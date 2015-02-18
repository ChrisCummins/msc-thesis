/*! \file operator_macros_cu.inl
 *  \brief Contains macro defintions for user functions using CUDA and CPU/OpenMP backend.
 */

/*!
 *  \ingroup userfunc
 *
 * \{
 */

/*! \def OVERLAP_DEF_FUNC(name, type1)
 *
 *  Macro defintion for user functions that is specified when user function is not mandatory for computation.
 *  The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can be used by the MapOverlap skeleton for applying 2D non-separable overlap.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 */
#define OVERLAP_DEF_FUNC(name, type1)\
struct name\
{\
	typedef type1 TYPE;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::OVERLAP;\
    }\
    type1 CPU(type1 param1)\
    {\
        return param1;\
    }\
    __device__ type1 CU(type1 param1)\
    {\
        return param1;\
    }\
};


/*! \def UNARY_FUNC(name, type1, param1, func)
 *
 *  Macro defintion for Unary user functions. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can be used by the Map and MapReduce skeletons.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param param1 Name of parameter one.
 *  \param func Function body.
 */
#define UNARY_FUNC(name, type1, param1, func)\
struct name\
{\
    typedef type1 TYPE;\
    bool isConst;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::UNARY;\
        isConst = true;\
    }\
    type1 dummy;\
    type1 getConstant() {return dummy;}\
    type1 CPU(type1 param1)\
    {\
        func\
    }\
    type1 CPU(type1 param1, type1 dummy)\
    {\
        return dummy;\
    }\
    type1 CPU(type1 param1, type1 dummy, type1 dummy2)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1)\
    {\
        func\
    }\
    __device__ type1 CU(type1 param1, type1 dummy)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 dummy, type1 dummy2)\
    {\
        return dummy;\
    }\
};

/*! \def UNARY_FUNC_CONSTANT(name, type1, constType, param1, const1, func)
 *
 *  Macro defintion for Unary user functions which also uses a constant. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can be used by the Map and MapReduce skeletons.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param constType Type of constant.
 *  \param param1 Name of parameter one.
 *  \param const1 Name of a constant which can be used in the body.
 *  \param func Function body.
 */
#define UNARY_FUNC_CONSTANT(name, type1, constType, param1, const1, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef constType CONST_TYPE;\
	bool isConst;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::UNARY;\
        isConst = true;\
        const1 = CONST_TYPE();\
    }\
    constType const1;\
    constType getConstant() {return const1;}\
    void setConstant(constType _v)\
    {\
    	const1 = _v;\
    }\
    type1 CPU(type1 param1)\
    {\
        func\
    }\
    type1 CPU(type1 param1, type1 dummy)\
    {\
        return dummy;\
    }\
    type1 CPU(type1 param1, type1 dummy, type1 dummy2)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1)\
    {\
        func\
    }\
    __device__ type1 CU(type1 param1, type1 dummy)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 dummy, type1 dummy2)\
    {\
        return dummy;\
    }\
};

/*! \def BINARY_FUNC(name, type1, param1, param2, func)
 *
 *  Macro defintion for Binary user functions. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can be used by the Map, Reduce and MapReduce skeletons.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param param1 Name of parameter one.
 *  \param param2 Name of parameter two.
 *  \param func Function body.
 */
#define BINARY_FUNC(name, type1, param1, param2, func)\
struct name\
{\
	typedef type1 TYPE;\
	bool isConst;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::BINARY;\
        isConst = false;\
    }\
    type1 dummy;\
    type1 getConstant() {return dummy;}\
    type1 CPU(type1 param1, type1 param2)\
    {\
        func\
    }\
    type1 CPU(type1 dummy)\
    {\
        return dummy;\
    }\
    type1 CPU(type1 param1, type1 dummy, type1 dummy2)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 param2)\
    {\
        func\
    }\
    __device__ type1 CU(type1 dummy)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 dummy, type1 dummy2)\
    {\
        return dummy;\
    }\
};

/*! \def BINARY_FUNC_CONSTANT(name, type1, constType, param1, param2, const1, func)
 *
 *  Macro defintion for Binary user functions which also uses a constant. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can be used by the Map and MapReduce skeletons.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param constType Type of constant.
 *  \param param1 Name of parameter one.
 *  \param param2 Name of parameter two.
 *  \param const1 Name of a constant which can be used in the body.
 *  \param func Function body.
 */
#define BINARY_FUNC_CONSTANT(name, type1, constType, param1, param2, const1, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef constType CONST_TYPE;\
	bool isConst;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::BINARY;\
        isConst = true;\
        const1 = CONST_TYPE();\
    }\
    constType const1;\
    constType getConstant() {return const1;}\
    void setConstant(constType _v)\
    {\
    	const1 = _v;\
    }\
    type1 CPU(type1 param1, type1 param2)\
    {\
        func\
    }\
    type1 CPU(type1 dummy)\
    {\
        return dummy;\
    }\
    type1 CPU(type1 param1, type1 dummy, type1 dummy2)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 param2)\
    {\
        func\
    }\
    __device__ type1 CU(type1 dummy)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 dummy, type1 dummy2)\
    {\
        return dummy;\
    }\
};

/*! \def TERNARY_FUNC(name, type1, param1, param2, param3, func)
 *
 *  Macro defintion for Trinary user functions. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can be used by the Map and MapReduce skeletons.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param param1 Name of parameter one.
 *  \param param2 Name of parameter two.
 *  \param param3 Name of parameter three.
 *  \param func Function body.
 */
#define TERNARY_FUNC(name, type1, param1, param2, param3, func)\
struct name\
{\
	typedef type1 TYPE;\
	bool isConst;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::TERNARY;\
        isConst = false;\
    }\
    type1 dummy;\
    type1 getConstant() {return dummy;}\
    type1 CPU(type1 param1, type1 param2, type1 param3)\
    {\
        func\
    }\
    type1 CPU(type1 dummy)\
    {\
        return dummy;\
    }\
    type1 CPU(type1 param1, type1 dummy)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 param2, type1 param3)\
    {\
        func\
    }\
    __device__ type1 CU(type1 dummy)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 dummy)\
    {\
        return dummy;\
    }\
};

/*! \def TERNARY_FUNC_CONSTANT(name, type1, constType, param1, param2, param3, const1, func)
 *
 *  Macro defintion for Trinary user functions which also uses a constant. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can be used by the Map and MapReduce skeletons.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param constType Type of constant.
 *  \param param1 Name of parameter one.
 *  \param param2 Name of parameter two.
 *  \param param3 Name of parameter three.
 *  \param const1 Name of a constant which can be used in the body.
 *  \param func Function body.
 */
#define TERNARY_FUNC_CONSTANT(name, type1, constType, param1, param2, param3, const1, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef constType CONST_TYPE;\
	bool isConst;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::TERNARY;\
        isConst = true;\
        const1 = CONST_TYPE();\
    }\
    constType const1;\
    constType getConstant() {return const1;}\
    void setConstant(constType _v)\
    {\
    	const1 = _v;\
    }\
    type1 CPU(type1 param1, type1 param2, type1 param3)\
    {\
        func\
    }\
    type1 CPU(type1 dummy)\
    {\
        return dummy;\
    }\
    type1 CPU(type1 param1, type1 dummy)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 param2, type1 param3)\
    {\
        func\
    }\
    __device__ type1 CU(type1 dummy)\
    {\
        return dummy;\
    }\
    __device__ type1 CU(type1 param1, type1 dummy)\
    {\
        return dummy;\
    }\
};

/*! \def OVERLAP_FUNC(name, type1, over, param1, func)
 *
 *  Macro defintion for Overlap user functions. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used for MapOverlap skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param over The overlap length used by the function.
 *  \param param1 Name of parameter one.
 *  \param func Function body.
 */
#define OVERLAP_FUNC(name, type1, over, param1, func)\
struct name\
{\
	typedef type1 TYPE;\
    size_t overlap;\
    size_t stride;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::OVERLAP;\
        overlap = over;\
		stride = 1;\
    }\
    size_t getStride() {return stride;}\
    void setStride(size_t _v) {stride = _v;}\
    type1 CPU(type1 * param1)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1)\
    {\
	    size_t stride=1;\
        func\
    }\
};

/*! \def OVERLAP_FUNC_STR(name, type1, over, param1, stride, func)
 *
 *  Macro defintion for Overlap user functions with support for strided access in body.
 *  Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used for MapOverlap skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param over The overlap length used by the function.
 *  \param param1 Name of parameter one.
 *  \param stride the stride which is used to access items column-wise.
 *  \param func Function body.
 */
#define OVERLAP_FUNC_STR(name, type1, over, param1, stride, func)\
struct name\
{\
    typedef type1 TYPE;\
    size_t overlap;\
    size_t stride;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::OVERLAP;\
        overlap = over;\
	stride = 1;\
    }\
    size_t getStride() {return stride;}\
    void setStride(size_t _v) {stride = _v;}\
    type1 CPU(type1 * param1)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1)\
    {\
    	size_t stride=1;\
        func\
    }\
};



/*! \def OVERLAP_FUNC_2D_STR(name, type1, overX, overY, param1, stride, func)
 *
 *  Macro defintion for 2D Overlap user functions with support for strided access in function body.
 *  Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used with 2DMapOverlap skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param overX The overlap length on horizontal axis used by the function.
 *  \param overY The overlap length on vertical axis used by the function.
 *  \param param1 Name of parameter one.
 *  \param stride the stride which is used to access items column-wise.
 *  \param func Function body.
 */
#define OVERLAP_FUNC_2D_STR(name, type1, overX, overY, param1, stride, func)\
struct name\
{\
	typedef type1 TYPE;\
    size_t overlapX;\
    size_t overlapY;\
    size_t stride;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::OVERLAP_2D;\
        overlapX = overX;\
        overlapY = overY;\
		stride = 1;\
    }\
    size_t getStride() {return stride;}\
    void setStride(size_t _v) {stride = _v;}\
    type1 CPU(type1 * param1)\
    {\
        func\
    }\
    type1 CPU(type1 * param1, size_t stride)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1, size_t stride)\
    {\
        func\
    }\
};

/*! \def ARRAY_FUNC(name, type1, param1, param2, func)
 *
 *  Macro defintion for Array user functions. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used by the MapArray skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param param1 Name of parameter one. Can be accessed as an array in the body.
 *  \param param2 Name of parameter two. Only one element is accessible in the body.
 *  \param func Function body.
 */
#define ARRAY_FUNC(name, type1, param1, param2, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef type1 CONST_TYPE;\
    skepu::FuncType funcType;\
    type1 dummy;\
    type1 getConstant() {return dummy;}\
    name()\
    {\
        funcType = skepu::ARRAY;\
    }\
    type1 CPU(type1 * param1, type1 param2)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1, type1 param2)\
    {\
        func\
    }\
};


/*! \def ARRAY_FUNC(name, type1, constType, param1, param2, const1, func)
 *
 *  Macro defintion for Array user functions. Includes only a CPU variant also used for OpenMP.
 *  The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used by the MapArray skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param constType Type of constant.
 *  \param param1 Name of parameter one. Can be accessed as an array in the body.
 *  \param param2 Name of parameter two. Only one element is accessible in the body.
 *  \param const1 Name of constant parameter.
 *  \param func Function body.
 */
#define ARRAY_FUNC_CONSTANT(name, type1, constType, param1, param2, const1, func)\
struct name\
{\
    typedef type1 TYPE;\
    typedef constType CONST_TYPE;\
    skepu::FuncType funcType;\
    constType const1;\
    constType getConstant() {return const1;}\
    void setConstant(constType _v)\
    {\
    	const1 = _v;\
    }\
    name()\
    {\
        funcType = skepu::ARRAY;\
    }\
    inline type1 CPU(type1 * param1, type1 param2)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1, type1 param2)\
    {\
        func\
    }\
};





/*! \def ARRAY_FUNC_MATR(name, type1, param1, param2, xindex, yindex, func)
 *
 *  Macro defintion for Array user functions for Matrix. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used by the MapArray skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param param1 Name of parameter one. Can be accessed as an array in the body.
 *  \param param2 Name of parameter two. Only one element is accessible in the body.
 *  \param xindex Index of param2 on x-axis.
 *  \param yindex Index of param2 on y-axis.
 *  \param func Function body.
 */
#define ARRAY_FUNC_MATR(name, type1, param1, param2, xindex, yindex, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef type1 CONST_TYPE;\
    skepu::FuncType funcType;\
    type1 dummy;\
    type1 getConstant() {return dummy;}\
    name()\
    {\
        funcType = skepu::ARRAY_INDEX;\
    }\
    type1 CPU(type1 * param1, type1 param2, size_t xindex, size_t yindex)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1, type1 param2, size_t xindex, size_t yindex)\
    {\
        func\
    }\
};


/*! \def ARRAY_FUNC_MATR_BLOCK_WISE(name, type1, param1, param2, p2BlockSize, func)
 *
 *  Macro defintion for Block-wise Array user functions for Matrix. Includes both
 *  a CUDA variant (__device__ function) and a CPU variant also used for OpenMP.
 *  The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used by the MapArray skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param param1 Name of parameter one. Can access all elements of param1 as an array in the body.
 *  \param param2 Name of parameter two. Can access "p2BlockSize" elements of param2 as an array in the body.
 *  \param p2BlockSize The size of param2 block. i.e., number of elements that can be accessed from param2.
 *  \param func Function body.
 */
#define ARRAY_FUNC_MATR_BLOCK_WISE(name, type1, param1, param2, p2BlockSize, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef type1 CONST_TYPE;\
    skepu::FuncType funcType;\
    size_t param2BlockSize;\
    type1 dummy;\
    type1 getConstant() {return dummy;}\
    name()\
    {\
        funcType = skepu::ARRAY_INDEX_BLOCK_WISE;\
        param2BlockSize = p2BlockSize;\
    }\
    type1 CPU(type1 * param1, type1 * param2)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1, type1 * param2)\
    {\
        func\
    }\
};






/*! \def ARRAY_FUNC_SPARSE_MATR_BLOCK_WISE(name, type1, param1, param2, p2BlockSize, func)
 *
 *  Macro defintion for Block-wise Array user functions for SparseMatrix. Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP.
 *  The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used by the MapArray skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param param1 Name of parameter one. Can access all elements of param1 as an array in the body.
 *  \param param2 Name of parameter two. Can access "p2BlockSize" elements of param2 as an array in the body.
 *  \param local_nnz The number of non-zero elements in param2. "local_nnz<=p2BlockSize" always.
 *  \param p1Idx Index of param1 elements corresponding to non-zero elements found in param2.
 *  \param p2BlockSize The size of param2 block. i.e., number of elements that can be accessed from param2.
 *  \param func Function body.
 */
#define ARRAY_FUNC_SPARSE_MATR_BLOCK_WISE(name, type1, param1, param2, local_nnz, p1Idx, p2BlockSize, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef type1 CONST_TYPE;\
    skepu::FuncType funcType;\
    size_t param2BlockSize;\
    type1 dummy;\
    type1 getConstant() {return dummy;}\
    name()\
    {\
        funcType = skepu::ARRAY_INDEX_SPARSE_BLOCK_WISE;\
        param2BlockSize = p2BlockSize;\
    }\
    type1 CPU(type1 * param1, type1 * param2, size_t local_nnz, size_t * p1Idx)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1, type1 * param2, size_t local_nnz, size_t * p1Idx)\
    {\
        func\
    }\
};

/*! \def ARRAY_FUNC_MATR_CONSTANT(name, type1, constType, param1, param2, const1, xindex, yindex, func)
 *
 *  Macro defintion for Array user functions for Matrix which also supports two constants.
 *  Includes both a CUDA variant (__device__ function)
 *  and a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used by the MapArray skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param constType Type of constant.
 *  \param param1 Name of parameter one. Can be accessed as an array in the body.
 *  \param param2 Name of parameter two. Only one element is accessible in the body.
 *  \param const1 Name of constant one which can be used in the body.
 *  \param xindex Index of param2 on x-axis.
 *  \param yindex Index of param2 on y-axis.
 *  \param func Function body.
 */
#define ARRAY_FUNC_MATR_CONSTANT(name, type1, constType, param1, param2, const1, xindex, yindex, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef constType CONST_TYPE;\
    skepu::FuncType funcType;\
    constType const1;\
    constType getConstant() {return const1;}\
    void setConstant(constType _v)\
    {\
    	const1 = _v;\
    }\
    name()\
    {\
        funcType = skepu::ARRAY_INDEX;\
        const1 = CONST_TYPE();\
    }\
    type1 CPU(type1 * param1, type1 param2, size_t xindex, size_t yindex)\
    {\
        func\
    }\
    __device__ type1 CU(type1 * param1, type1 param2, size_t xindex, size_t yindex)\
    {\
        func\
    }\
};


/*! \def GENERATE_FUNC(name, type1, constType, index, const1, func)
 *
 *  Macro defintion for Generate user functions. Includes both a CUDA variant (__device__ function) and
 *  a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used by the Generate skeleton.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param constType Type of constant.
 *  \param index Name of the index variable which will hold the index of the value to be generated.
 *  \param const1 Name of a constant which can be used in the body.
 *  \param func Function body.
 */
#define GENERATE_FUNC(name, type1, constType, index, const1, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef constType CONST_TYPE;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::GENERATE;\
        const1 = CONST_TYPE();\
    }\
    constType const1;\
    constType getConstant() {return const1;}\
    void setConstant(constType _v) {const1 = _v;}\
    type1 CPU(size_t index)\
    {\
        func\
    }\
    __device__ type1 CU(size_t index)\
    {\
        func\
    }\
};


/*! \def GENERATE_FUNC_MATRIX(name, type1, constType, xindex, yindex, const1, func)
 *
 *  Macro defintion for Generate user functions for matrix objects. Includes both a CUDA variant (__device__ function) and
 *  a CPU variant also used for OpenMP. The defintion expands as a Struct which can
 *  be used when creating new skeletons. Can only be used by the Generate skeleton with matrix objects.
 *
 *  \param name Function name.
 *  \param type1 Type of function parameters.
 *  \param constType Type of constant.
 *  \param xindex Name of the column index variable which will hold the (x-axis) column index of the value to be generated.
 *  \param yindex Name of the row index variable which will hold the (y-axis) row index of the value to be generated.
 *  \param const1 Name of a constant which can be used in the body.
 *  \param func Function body.
 */
#define GENERATE_FUNC_MATRIX(name, type1, constType, xindex, yindex, const1, func)\
struct name\
{\
	typedef type1 TYPE;\
	typedef constType CONST_TYPE;\
    skepu::FuncType funcType;\
    name()\
    {\
        funcType = skepu::GENERATE_MATRIX;\
        const1 = CONST_TYPE();\
    }\
    constType const1;\
    constType getConstant() {return const1;}\
    void setConstant(constType _v) {const1 = _v;}\
    type1 CPU(size_t xindex, size_t yindex)\
    {\
        func\
    }\
    __device__ type1 CU(size_t xindex, size_t yindex)\
    {\
        func\
    }\
};

/*!
 *  \}
 */

