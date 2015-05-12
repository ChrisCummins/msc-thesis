#define skelcl_get_device_id() 0




#ifndef float_MATRIX_T
typedef struct {
  __global float* data;
  unsigned int col_count;
} float_matrix_t;
#define float_MATRIX_T
#endif

#ifndef MATRIX_GET
#define get(m, y, x) m.data[(int)((y) * m.col_count + (x))]
#define MATRIX_GET
#endif
#ifndef MATRIX_SET
#define set(m, y, x, v) m.data[(int)((y) * m.col_count + (x))] = (v)
#define MATRIX_SET
#endif
#ifndef float_MATRIX_T
typedef struct {
  __global float* data;
  unsigned int col_count;
} float_matrix_t;
#define float_MATRIX_T
#endif

#ifndef MATRIX_GET
#define get(m, y, x) m.data[(int)((y) * m.col_count + (x))]
#define MATRIX_GET
#endif
#ifndef MATRIX_SET
#define set(m, y, x, v) m.data[(int)((y) * m.col_count + (x))] = (v)
#define MATRIX_SET
#endif
#define SCL_NORTH (0)
#define SCL_WEST  (0)
#define SCL_SOUTH (0)
#define SCL_EAST  (0)
#define SCL_TILE_WIDTH  (get_local_size(0) + SCL_WEST + SCL_EAST)
#define SCL_TILE_HEIGHT (get_local_size(1) + SCL_NORTH + SCL_SOUTH)
#define SCL_COL   (get_global_id(0))
#define SCL_ROW   (get_global_id(1))
#define SCL_L_COL (get_local_id(0))
#define SCL_L_ROW (get_local_id(1))
#define SCL_L_COL_COUNT (get_local_size(0))
#define SCL_L_ROW_COUNT (get_local_size(1))
#define SCL_L_ID (SCL_L_ROW * SCL_L_COL_COUNT + SCL_L_COL)
#define SCL_ROWS (SCL_ELEMENTS / SCL_COLS)


    typedef float SCL_TYPE_0;
    typedef float SCL_TYPE_1;

    typedef struct {
        __global SCL_TYPE_1* data;
    } input_matrix_t;

    //In case, local memory is used
    SCL_TYPE_1 getData(input_matrix_t* matrix, int x, int y){
        return matrix->data[0];
    }

float USR_FUNC(input_matrix_t* img, __global float* kernelVec, int range)
{
    float val = getData(img, 0, 0);
    if(val <= 20) {
        val = 0;
    } else if(val > 20 && val < 40) {
        val = 0;
    } else {
        val = 255;
    }
    return val;
}
