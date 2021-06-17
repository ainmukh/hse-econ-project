#define PY_SSIZE_T_CLEAN
#include "Python.h"
//#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "approx_class.h"
#include <Eigen/Core>

using Eigen::VectorXd;

/*
 * approx_class.h
 *
 * Each function of the form type_cdf defines the
 * CDF function for a different numpy dtype.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 *
 */

static PyMethodDef CDFUfuncMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definitions must precede the PyMODINIT_FUNC. */

static void double_cdf(char **args, npy_intp *dimensions,
                       npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp j;
    npy_intp n = dimensions[1];  // x size
    npy_intp m = dimensions[2];  // alphas size

    char *x = args[0];  // x
    char *alpha = args[1];
    char *beta = args[2];
    char *alphas = args[3];
    char *out = args[4];
    npy_intp in_step = steps[5], alpha_step = steps[6], out_step = steps[7];

    double loc = *(double *) alpha;
    double scale = *(double *) beta;

    VectorXd param(m);
    for (j = 0; j < m; j++) {
        param[j] = *(double *)alphas;
        alphas += alpha_step;
    }

    Distribution<double> distribution(loc, scale, param);

    double tmp;
#pragma omp parallel for firstprivate(distribution)
    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(double *)(x + i * in_step);
        *(( double *)(out + out_step * i)) = distribution.cdf(tmp);
        /*END main ufunc computation*/
    }
}

/*This gives pointers to the above functions*/
PyUFuncGenericFunction funcs[1] = {reinterpret_cast<PyUFuncGenericFunction>(&double_cdf)};

static char types[5] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *data[1] = {NULL};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "cdf_ufunc",
        NULL,
        -1,
        CDFUfuncMethods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_cdf_ufunc(void)
{
    PyObject *m, *cdf, *d;

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    cdf = PyUFunc_FromFuncAndDataAndSignature(funcs, data, types, 1, 4, 1,
                                              PyUFunc_None, "cdf",
                                              "cdf_docstring", 0, "(n),(),(),(m)->(n)");

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "cdf", cdf);
    Py_DECREF(cdf);

    return m;
}
