#define skelcl_get_device_id() 0


typedef struct { float x; float y; float z; float w; } data_t;

#ifndef data_t_MATRIX_T
typedef struct {
  __global data_t* data;
  unsigned int col_count;
} data_t_matrix_t;
#define data_t_MATRIX_T
#endif

#ifndef MATRIX_GET
#define get(m, y, x) m.data[(int)((y) * m.col_count + (x))]
#define MATRIX_GET
#endif
#ifndef MATRIX_SET
#define set(m, y, x, v) m.data[(int)((y) * m.col_count + (x))] = (v)
#define MATRIX_SET
#endif
#ifndef data_t_MATRIX_T
typedef struct {
  __global data_t* data;
  unsigned int col_count;
} data_t_matrix_t;
#define data_t_MATRIX_T
#endif

#ifndef MATRIX_GET
#define get(m, y, x) m.data[(int)((y) * m.col_count + (x))]
#define MATRIX_GET
#endif
#ifndef MATRIX_SET
#define set(m, y, x, v) m.data[(int)((y) * m.col_count + (x))] = (v)
#define MATRIX_SET
#endif
#define NEUTRAL {0.000000e+00f,0.000000e+00f,0.000000e+00f,0.000000e+00f}
#define SCL_NORTH (0)
#define SCL_WEST  (0)
#define SCL_SOUTH (1)
#define SCL_EAST  (1)
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


typedef data_t SCL_TYPE_0;
typedef data_t SCL_TYPE_1;

typedef struct {
    __local SCL_TYPE_1* data;
} input_matrix_t;

//In case, local memory is used
SCL_TYPE_1 getData(input_matrix_t* matrix, int x, int y){
    int offsetNorth = SCL_NORTH * SCL_TILE_WIDTH;
    int currentIndex = SCL_L_ROW * SCL_TILE_WIDTH + SCL_L_COL;
    int shift = x - y * SCL_TILE_WIDTH;

    return matrix->data[currentIndex+offsetNorth+shift+SCL_WEST];
}

#define Pr 1.000000e+13f
#define eps_r 4.000000e+00f
#define eps_b 1.000000e+00f
#define abs_cell_size 0
#define array_size 1024
#define array_size_2 512
#define log_2 6.931472e-01f
#define pi 3.141593e+00f
#define dt_r 1.667821e-15f
#define k 1.192011e-03f
#define a1 1.580421e+06f
#define a2 -1.530277e-01f
#define omega_a2 1.421223e+31f
#define Nges 3.311000e+24f
#define dt_tau10 1.667821e-03f
#define dt_tau32 1.667821e-02f
#define dt_tau21 1.667820e-05f
#define sqrt_eps0_mu0 2.654419e-03f
#define c 2.997924e+08f
#define src_x 50
#define src_y 50
#define idx_x (get_global_id(1))
#define idx_y (get_global_id(0))

#define __sq(x) ((x)*(x))
#define __cu(x) ((x)*(x)*(x))

#define fx pml(idx_y+1, abs_cell_size, array_size_2, 0.25f)
#define fy pml(idx_x+1, abs_cell_size, array_size_2, 0.25f)
#define gx pml(idx_y+1, abs_cell_size, array_size_2, 0.33f)
#define gy pml(idx_x+1, abs_cell_size, array_size_2, 0.33f)


data_t pml(int i, int acs, int as, float scale)
{
	//		  1		   2	    3	   unused
	data_t ret = { 0.0f, 1.0f, 1.0f, 0.0f };
	if (i > as)
		i = (as * 2) - 1 - i;

	if (i >= acs)
		return ret;

	float xnum = acs - i;
	float xn = 0;

	xnum = (xnum - 0.5f) / acs;
	xn = scale * xnum * xnum * xnum;
	ret.x = xn;
	ret.y = 1.0f / (1.0f + xn);
	ret.z = (1.0f - xn) / (1.0f + xn);

	return ret;
}

float jsrc(float t)
{
	if ((t < (3.0f)))
		return 1000.f * exp(-2.0f * log_2 * __sq((t - 0.0f) / 0.5f)) * sin(2.0f * pi * 2.0f * t);

	return 0.0f;
}


//	   y - 1		 x - 1
data_t eFieldEquation(input_matrix_t* mH, data_t_matrix_t mE, float t)
{
	data_t E = get(mE, idx_x, idx_y);

	data_t cx = gx;
	data_t cy = gy;

	data_t H  = getData(mH, 0, 0);
	data_t Ha = getData(mH, 0, 1);
	data_t Hc = getData(mH, -1, 0);

	E.y = cx.z * cy.z * E.y + cx.y * cy.y * 0.5f * (H.y - Hc.y - H.x + Ha.x);

	if ((idx_x == src_x) && (idx_y == src_y))
		E.y += sqrt_eps0_mu0 * jsrc(t);

	float eps = eps_b;
	if ((fx.x + fy.x) == 0.0f)
	{
		if (E.w != 0.0f)
			eps = eps_r; // in particle
	}

	E.z = (E.y - c * E.x) * (1.0f / eps);

	return E;
}


data_t USR_FUNC(input_matrix_t* mE, data_t_matrix_t mH)
{
	data_t H = get(mH, idx_x, idx_y);

	data_t E  = getData(mE, 0, 0);
	data_t Ea = getData(mE, 0, -1);
	data_t Ec = getData(mE, 1, 0);

	float2 curl_e = { 0.0f, 0.0f };

	data_t cx = fx;
	data_t cy = fy;

	curl_e.x = E.z - Ea.z;
	curl_e.y = Ec.z - E.z;

	if (cx.x > 0.0f)
	{
		H.z = H.z + cx.x * curl_e.x;
	}
	if (cy.x > 0.0f)
	{
		H.w = H.w + cy.x * curl_e.y;
	}

	H.x = cy.z * H.x + cy.y * 0.5f * (curl_e.x + H.z);
	H.y = cx.z * H.y + cx.y * 0.5f * (curl_e.y + H.w);
	return H;
}

#define STENCIL_PADDING_NEUTRAL         1
#define STENCIL_PADDING_NEAREST         0
#define STENCIL_PADDING_NEAREST_INITIAL 0


// The three different padding types affect the values loaded into the border
// regions. By defining macros to determine which value to return, we can
// save on a huge amount of conditional logic between the different padding
// types.
#if STENCIL_PADDING_NEAREST_INITIAL
#  define input(y, x)  SCL_IN[(y) * SCL_COLS + (x)]
#  define border(y, x) SCL_INITIAL[(y) * SCL_COLS + (x)]
#elif STENCIL_PADDING_NEAREST
#  define input(y, x)  SCL_IN[(y) * SCL_COLS + (x)]
#  define border(y, x) SCL_IN[(y) * SCL_COLS + (x)]
#elif STENCIL_PADDING_NEUTRAL
#  define input(y, x)  SCL_IN[(y) * SCL_COLS + (x)]
#  define border(y, x) neutral
#else
// Fall-through case, throw an error.
#  error Unrecognised padding type.
#endif

// Macro function to index flat SCL_LOCAL array.
#define local(y, x) SCL_LOCAL[(y) * SCL_TILE_WIDTH + (x)]

// Define a helper macro which accepts an \"a\" and \"b\" value, returning \"a\"
// if the padding type is neutral, else \"b\".
#if STENCIL_PADDING_NEUTRAL
#  define neutralPaddingIfElse(a, b) (a)
#else
#  define neutralPaddingIfElse(a, b) (b)
#endif
