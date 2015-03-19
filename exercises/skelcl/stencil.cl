double func(input_matrix_t *data, int range) {
    double sum = 0.0;
    int j, i;

    for (j = -range; j <= range; j++) {
        for (i = -range; i <= range; i++) {
            sum += getData(data, j, i);
        }
    }

    //return sum;
    return getData(data, 1, 0);
}
