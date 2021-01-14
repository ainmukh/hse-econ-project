/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

#include "Approximation_alex.h"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <iostream>


/*  wrapped cdf function */
//static PyObject* cos_func_np(PyObject* self, PyObject* args)
static PyObject* cdf_func(PyObject* self, PyObject* args)
{
    PyArrayObject *arrays[2];  /* holds input and output array */
    PyObject *ret;
    NpyIter *iter;
    npy_uint32 op_flags[2];
    npy_uint32 iterator_flags;
    PyArray_Descr *op_dtypes[2];

    NpyIter_IterNextFunc *iternext;

    double alpha;
    double beta;
    PyArrayObject* alphas;
    PyArrayObject* theta;
    PyArrayObject* x;

    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "ddO!O!O!", &alpha, &beta, &PyArray_Type, &alphas, &PyArray_Type, &theta, &PyArray_Type, &x)) {
        return NULL;
    }
    if (PyArray_NDIM(alphas) != 1) {
        PyErr_Format(PyExc_ValueError,"Expected 1d alphas, found %d", PyArray_NDIM(alphas));
        return NULL;
    }
    if (PyArray_NDIM(theta) != 1) {
        PyErr_Format(PyExc_ValueError,"Expected 1d theta, found %d", PyArray_NDIM(theta));
        return NULL;
    }
    if (PyArray_NDIM(x) != 1) {
        PyErr_Format(PyExc_ValueError,"Expected 1d x, found %d", PyArray_NDIM(x));
        return NULL;
    }
    if (PyArray_TYPE(alphas) != NPY_DOUBLE) {
        PyErr_Format(PyExc_ValueError,"Wrong alphas type: expected double, found %d", PyArray_TYPE(alphas));
        return NULL;
    }
    if (PyArray_TYPE(theta) != NPY_DOUBLE) {
        PyErr_Format(PyExc_ValueError,"Wrong theta type: expected double, found %d", PyArray_TYPE(theta));
        return NULL;
    }
    if (PyArray_TYPE(x) != NPY_DOUBLE) {
        PyErr_Format(PyExc_ValueError,"Wrong x type: expected double, found %d", PyArray_TYPE(x));
        return NULL;
    }

    double* alphas_ = reinterpret_cast<double*>(PyArray_DATA(alphas));
    double* theta_ = reinterpret_cast<double*>(PyArray_DATA(theta));
    double* x_ = reinterpret_cast<double*>(PyArray_DATA(x));
//    auto[func, params, result_size] = CDF_func(alpha, beta, alphas_, theta_, x_, PyArray_DIMS(alphas)[0], PyArray_DIMS(theta)[0], PyArray_DIMS(x)[0]);
    auto[fx, dfx, result_size] = CDF_func(alpha, beta, alphas_, theta_, x_, PyArray_DIMS(alphas)[0], PyArray_DIMS(theta)[0], PyArray_DIMS(x)[0]);
    npy_intp dims[] = {static_cast<npy_intp>(result_size)};
    PyObject* dfx_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<void*>(dfx));
    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(dfx_np), NPY_ARRAY_OWNDATA);

    return Py_BuildValue("dO", fx, dfx_np);
}


/*  define functions in module */
//static PyMethodDef CosMethods[] =
//{
//     {"cos_func_np", cos_func_np, METH_VARARGS,
//         "evaluate the cosine on a numpy array"},
//     {NULL, NULL, 0, NULL}
//};
static PyMethodDef CDFMethods[] =
{
     {"CDF", cdf_func, METH_VARARGS,
         "evaluate the CDF on a numpy array"},
     {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
//static struct PyModuleDef cModPyDem = {
//    PyModuleDef_HEAD_INIT,
//    "cos_module", "Some documentation",
//    -1,
//    CosMethods
//};
static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "CDF_module", "Some documentation",
        -1,
        CDFMethods
};

//PyMODINIT_FUNC PyInit_cos_module_np(void) {
//    PyObject *module;
//    module = PyModule_Create(&cModPyDem);
//    if(module==NULL) return NULL;
//    /* IMPORTANT: this must be called */
//    import_array();
//    if (PyErr_Occurred()) return NULL;
//    return module;
//}
PyMODINIT_FUNC PyInit_cdf_module_np(void) {
    PyObject *module;
    module = PyModule_Create(&cModPyDem);
    if(module==NULL) return NULL;
    /* IMPORTANT: this must be called */
    import_array();
    if (PyErr_Occurred()) return NULL;
    return module;
}

#else

#ERROR Trying to compile as python2

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC initcdf_module_np(void) {
    PyObject *module;
    module = Py_InitModule("cdf_module_np", CDFMethods);
    if(module==NULL) return;
    /* IMPORTANT: this must be called */
    import_array();
    return;
}

#endif
