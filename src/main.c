#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <mpi.h>

#define NUM_FEATURES 4
#define NUM_COLUMNS 5
#define NUM_CLASSES 3
#define MAX_LINE_LENGTH 1024

typedef struct {
    int feature;
    double threshold;
    int left_prediction;
    int right_prediction;
    double gini;
} SplitNode;

/* Count valid data rows in the CSV file, excluding the header line. */
int count_rows(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) return -1;

    char line[MAX_LINE_LENGTH];
    int count = 0;

    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) <= 1) continue;
        count++;
    }

    fclose(file);
    return count;
}

/* Convert textual Iris class labels into integer class identifiers. */
int map_label(char* label) {
    label[strcspn(label, "\r\n")] = 0;

    if (strcmp(label, "Iris-setosa") == 0) return 0;
    if (strcmp(label, "Iris-versicolor") == 0) return 1;
    if (strcmp(label, "Iris-virginica") == 0) return 2;

    return -1;
}

const char* class_name(int label) {
    if (label == 0) return "Iris-setosa";
    if (label == 1) return "Iris-versicolor";
    if (label == 2) return "Iris-virginica";
    return "Unknown";
}

/* Compute Gini impurity: lower values indicate purer class distributions. */
double calculate_gini(int class_counts[NUM_CLASSES], int total_samples) {
    if (total_samples == 0) return 0.0;

    double gini = 1.0;

    for (int i = 0; i < NUM_CLASSES; i++) {
        double p = (double)class_counts[i] / total_samples;
        gini -= p * p;
    }

    return gini;
}

int majority_class(int counts[NUM_CLASSES]) {
    int best = 0;

    for (int c = 1; c < NUM_CLASSES; c++) {
        if (counts[c] > counts[best]) {
            best = c;
        }
    }

    return best;
}

/*
 * Read the CSV dataset into a flat numeric array.
 * The first four columns are features; the final column is preprocessed
 * from a string label into an integer class value.
 */
double* read_dataset_flat(const char* filename, int total_rows) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) return NULL;

    double* data = (double*)malloc(total_rows * NUM_COLUMNS * sizeof(double));
    if (data == NULL) {
        fclose(file);
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    int row = 0;

    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file) && row < total_rows) {
        if (strlen(line) <= 1) continue;

        char* token = strtok(line, ",");
        int col = 0;

        while (token != NULL && col < NUM_COLUMNS) {
            if (col < NUM_FEATURES) {
                data[row * NUM_COLUMNS + col] = atof(token);
            } else {
                int label = map_label(token);
                data[row * NUM_COLUMNS + col] = (double)label;
            }

            token = strtok(NULL, ",");
            col++;
        }

        row++;
    }

    fclose(file);
    return data;
}

/*
 * Determine whether a sample belongs to the node currently being split.
 * The root contains all samples; the second-level node contains only samples
 * routed to one side of the root split.
 */
int sample_belongs_to_node(
    double* sample,
    int is_root,
    SplitNode root,
    int go_right_from_root
) {
    if (is_root) return 1;

    double value = sample[root.feature];

    if (go_right_from_root) {
        return value > root.threshold;
    } else {
        return value <= root.threshold;
    }
}

/*
 * Search for the best split for one decision tree node.
 * Each process evaluates its local partition, then MPI collectives combine
 * the local statistics into global statistics for the full dataset.
 */
SplitNode find_best_split_for_node(
    double* local_data,
    int local_rows,
    int is_root,
    SplitNode root,
    int go_right_from_root
) {
    SplitNode result;
    result.feature = -1;
    result.threshold = 0.0;
    result.left_prediction = 0;
    result.right_prediction = 0;
    result.gini = DBL_MAX;

    int local_node_count = 0;
    int global_node_count = 0;

    for (int i = 0; i < local_rows; i++) {
        double* sample = &local_data[i * NUM_COLUMNS];

        if (sample_belongs_to_node(sample, is_root, root, go_right_from_root)) {
            local_node_count++;
        }
    }

    /*
     * MPI_Allreduce sums the number of samples assigned to this node across
     * all processes. Every process needs the global node size because it is
     * used to compute weighted Gini values for candidate splits.
     */
    MPI_Allreduce(
        &local_node_count,
        &global_node_count,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    if (global_node_count == 0) {
        return result;
    }

    double local_min[NUM_FEATURES];
    double local_max[NUM_FEATURES];
    double global_min[NUM_FEATURES];
    double global_max[NUM_FEATURES];

    for (int f = 0; f < NUM_FEATURES; f++) {
        local_min[f] = DBL_MAX;
        local_max[f] = -DBL_MAX;
    }

    for (int i = 0; i < local_rows; i++) {
        double* sample = &local_data[i * NUM_COLUMNS];

        if (!sample_belongs_to_node(sample, is_root, root, go_right_from_root)) {
            continue;
        }

        for (int f = 0; f < NUM_FEATURES; f++) {
            double value = sample[f];

            if (value < local_min[f]) local_min[f] = value;
            if (value > local_max[f]) local_max[f] = value;
        }
    }

    /*
     * MPI_Allreduce combines local feature ranges into global feature ranges.
     * The minimum and maximum values define candidate thresholds that are
     * consistent across all processes.
     */
    MPI_Allreduce(local_min, global_min, NUM_FEATURES, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_max, global_max, NUM_FEATURES, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    /* Evaluate three candidate thresholds per feature and keep the split
       with the lowest weighted Gini impurity. */
    for (int f = 0; f < NUM_FEATURES; f++) {
        double min_value = global_min[f];
        double max_value = global_max[f];

        if (max_value - min_value < 1e-9) continue;

        double thresholds[3];
        thresholds[0] = min_value + 0.25 * (max_value - min_value);
        thresholds[1] = min_value + 0.50 * (max_value - min_value);
        thresholds[2] = min_value + 0.75 * (max_value - min_value);

        for (int t = 0; t < 3; t++) {
            int local_counts[2][NUM_CLASSES] = {0};
            int global_counts[2][NUM_CLASSES] = {0};

            double threshold = thresholds[t];

            for (int i = 0; i < local_rows; i++) {
                double* sample = &local_data[i * NUM_COLUMNS];

                if (!sample_belongs_to_node(sample, is_root, root, go_right_from_root)) {
                    continue;
                }

                int label = (int)sample[NUM_FEATURES];

                if (label < 0 || label >= NUM_CLASSES) continue;

                if (sample[f] <= threshold) {
                    local_counts[0][label]++;
                } else {
                    local_counts[1][label]++;
                }
            }

            /*
             * MPI_Allreduce sums the local left/right class counts for this
             * candidate split. The resulting global counts are returned to
             * every process so all ranks select the same best split.
             */
            MPI_Allreduce(
                local_counts,
                global_counts,
                2 * NUM_CLASSES,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD
            );

            int left_total = 0;
            int right_total = 0;

            for (int c = 0; c < NUM_CLASSES; c++) {
                left_total += global_counts[0][c];
                right_total += global_counts[1][c];
            }

            if (left_total == 0 || right_total == 0) continue;

            double left_gini = calculate_gini(global_counts[0], left_total);
            double right_gini = calculate_gini(global_counts[1], right_total);

            double weighted_gini =
                ((double)left_total / global_node_count) * left_gini +
                ((double)right_total / global_node_count) * right_gini;

            if (weighted_gini < result.gini) {
                result.feature = f;
                result.threshold = threshold;
                result.gini = weighted_gini;
                result.left_prediction = majority_class(global_counts[0]);
                result.right_prediction = majority_class(global_counts[1]);
            }
        }
    }

    return result;
}

/*
 * Predict using the fixed depth-2 tree:
 * the root split creates a left leaf and a right child; the right child
 * then creates two additional leaves.
 */
int predict_depth2(double* sample, SplitNode root, SplitNode right_child) {
    if (sample[root.feature] <= root.threshold) {
        return root.left_prediction;
    }

    if (right_child.feature == -1) {
        return root.right_prediction;
    }

    if (sample[right_child.feature] <= right_child.threshold) {
        return right_child.left_prediction;
    }

    return right_child.right_prediction;
}

int main(int argc, char** argv) {
    int rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    const char* filename = "dataset/iris_large.csv";

    int total_rows = 0;
    int num_features = NUM_FEATURES;
    double* full_data = NULL;

    if (rank == 0) {
        /* Rank 0 performs file I/O and preprocessing before distributing data. */
        total_rows = count_rows(filename);

        if (total_rows == -1) {
            printf("Error: Could not open dataset file: %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        full_data = read_dataset_flat(filename, total_rows);

        if (full_data == NULL) {
            printf("Error: Could not read dataset into memory.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Rank 0 read dataset successfully.\n");
        printf("Total rows: %d\n", total_rows);
        printf("Number of features: %d\n", num_features);
        printf("Number of processes: %d\n", comm_sz);
    }

    /*
     * MPI_Bcast broadcasts dataset metadata from rank 0 to all processes.
     * This project uses it so every rank knows the total row count and number
     * of features before memory allocation and data distribution.
     */
    MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_features, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base_rows = total_rows / comm_sz;
    int remainder = total_rows % comm_sz;

    int local_rows = base_rows;
    if (rank < remainder) local_rows++;

    int* sendcounts = NULL;
    int* displs = NULL;

    if (rank == 0) {
        sendcounts = (int*)malloc(comm_sz * sizeof(int));
        displs = (int*)malloc(comm_sz * sizeof(int));

        int offset = 0;

        for (int i = 0; i < comm_sz; i++) {
            int rows_for_proc = base_rows;
            if (i < remainder) rows_for_proc++;

            sendcounts[i] = rows_for_proc * NUM_COLUMNS;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    double* local_data = (double*)malloc(local_rows * NUM_COLUMNS * sizeof(double));

    if (local_data == NULL) {
        printf("Rank %d: Error allocating local data.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*
     * MPI_Scatterv distributes the flat dataset from rank 0 to all ranks.
     * It is used instead of MPI_Scatter because row counts may differ when
     * the dataset size is not evenly divisible by the number of processes.
     * The communicated data consists of each process's assigned rows.
     */
    MPI_Scatterv(
        full_data,
        sendcounts,
        displs,
        MPI_DOUBLE,
        local_data,
        local_rows * NUM_COLUMNS,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    printf("Rank %d received %d rows.\n", rank, local_rows);

    int local_class_counts[NUM_CLASSES] = {0};
    int global_class_counts[NUM_CLASSES] = {0};

    for (int i = 0; i < local_rows; i++) {
        int label = (int)local_data[i * NUM_COLUMNS + NUM_FEATURES];

        if (label >= 0 && label < NUM_CLASSES) {
            local_class_counts[label]++;
        }
    }

    /*
     * MPI_Allreduce sums per-process class counts into a global distribution.
     * All processes receive the result, which supports consistent impurity
     * calculations and reporting.
     */
    MPI_Allreduce(
        local_class_counts,
        global_class_counts,
        NUM_CLASSES,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("\nGlobal class distribution:\n");
        printf("Class 0 (%s): %d\n", class_name(0), global_class_counts[0]);
        printf("Class 1 (%s): %d\n", class_name(1), global_class_counts[1]);
        printf("Class 2 (%s): %d\n", class_name(2), global_class_counts[2]);

        double root_gini = calculate_gini(global_class_counts, total_rows);
        printf("Root node Gini impurity: %.6f\n", root_gini);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /* A dummy parent is passed when searching for the root split. */
    SplitNode dummy_root;
    dummy_root.feature = -1;
    dummy_root.threshold = 0.0;
    dummy_root.left_prediction = 0;
    dummy_root.right_prediction = 0;
    dummy_root.gini = 0.0;

    /* Build a depth-2 tree: first find the root, then split the right branch. */
    SplitNode root = find_best_split_for_node(
        local_data,
        local_rows,
        1,
        dummy_root,
        0
    );

    SplitNode right_child = find_best_split_for_node(
        local_data,
        local_rows,
        0,
        root,
        1
    );

    int local_correct = 0;
    int global_correct = 0;

    /* Each rank predicts labels for its local rows and counts correct results. */
    for (int i = 0; i < local_rows; i++) {
        double* sample = &local_data[i * NUM_COLUMNS];
        int true_label = (int)sample[NUM_FEATURES];

        int predicted = predict_depth2(sample, root, right_child);

        if (predicted == true_label) {
            local_correct++;
        }
    }

    /*
     * MPI_Reduce sums the local correct-prediction counts on rank 0.
     * This gives the global number of correctly classified samples, which is
     * divided by the total row count to compute training accuracy.
     */
    MPI_Reduce(
        &local_correct,
        &global_correct,
        1,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;

    if (rank == 0) {
        printf("\nDepth-2 Decision Tree:\n");

        printf("\nRoot split:\n");
        printf("Feature index: %d\n", root.feature);
        printf("Threshold: %.6f\n", root.threshold);
        printf("Weighted Gini: %.6f\n", root.gini);
        printf("Left prediction: Class %d (%s)\n", root.left_prediction, class_name(root.left_prediction));
        printf("Right branch continues to child node.\n");

        printf("\nRight child split:\n");
        printf("Feature index: %d\n", right_child.feature);
        printf("Threshold: %.6f\n", right_child.threshold);
        printf("Weighted Gini: %.6f\n", right_child.gini);
        printf("Left prediction: Class %d (%s)\n", right_child.left_prediction, class_name(right_child.left_prediction));
        printf("Right prediction: Class %d (%s)\n", right_child.right_prediction, class_name(right_child.right_prediction));

        double accuracy = (double)global_correct / total_rows;
        printf("\nTraining accuracy: %.4f\n", accuracy);

        printf("Execution time: %.6f seconds\n", execution_time);
    }

    free(local_data);

    if (rank == 0) {
        free(full_data);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
