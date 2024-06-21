#ifndef COX_REGRESSION_FIT_OBJECT_H
#define COX_REGRESSION_FIT_OBJECT_H

#include <vector>
#include <cmath>
#include "utils/parameter_handler.h"

class StepFunction {
public:
    std::vector<double> x;
    std::vector<double> y;

    StepFunction() {}

    StepFunction(const std::vector<double>& _x, const std::vector<double>& _y) {
        x = _x;
        y = _y;
    }

    bool operator ==(const StepFunction& other) const {
        if (x.size() != other.x.size())
            return false;
        for (int i = 0; i < x.size(); ++i)
            if (fabs(x[i] - other.x[i]) > 1e-6 || fabs(y[i] - other.y[i]) > 1e-6 )
                return false;
        return true;
    }

    inline bool operator!=(const StepFunction& other) const { return !(*this == other); }
};


class Fit {
public:
    double alpha;
    double offset;
    StepFunction func;
    std::vector<double> coefs;
    Fit() {}
    Fit(const StepFunction& _func) {
        func = _func;
    }
    bool operator ==(const Fit& other) const {
        if (func != other.func || fabs(alpha - other.alpha) > 1e-6 || fabs(offset - other.offset) > 1e-6)
            return false;
        if (coefs.size() != other.coefs.size())
            return false;
        for (int i = 0; i < coefs.size(); ++i)
            if (fabs(coefs[i] - other.coefs[i]) > 1e-6 )
                return false;
        return true;
    }

    inline bool operator!=(const Fit& other) const { return !(*this == other); }

    std::string ToString() const {
        if (coefs.size() == 0)
            return "";
        std::string sol = std::to_string(coefs[0]);
        for (int i = 1; i < coefs.size(); ++i) {
            sol += ",";
            sol += std::to_string(coefs[i]);
        }
        return sol;
    }
};

#endif //COX_REGRESSION_FIT_OBJECT_H
