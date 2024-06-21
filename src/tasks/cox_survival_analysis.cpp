#include "tasks/cox_survival_analysis.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include "tasks/cox_regression/coxnet.h"
#include "tasks/cox_regression/fit-object.h"
#include "tasks/cox_regression/coxnet_wrapper.h"
#include <algorithm>
#include <iostream>     // std::cout
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib>

namespace STreeD {
    const Fit CoxSurvivalAnalysis::worst_label = Fit();

    // Read the event (censoring = 0 or death = 1) from the file and the extra (continuous) data
    CSAData CSAData::ReadData(std::istringstream &iss, int num_extra_cols) {
        int ev;
        iss >> ev;
        std::vector<double>vars = std::vector<double>(num_extra_cols);
        for (int i = 0; i < num_extra_cols; ++i) {
            iss >> vars[i];
        }
        return {ev, vars};
    }

    std::vector<std::vector<double>> dotProduct(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
        assert(a[0].size() == b.size());
        int n = a.size();
        int m = a[0].size();
        int k = b[0].size();
        std::vector<std::vector<double>> c = std::vector<std::vector<double>>(n, std::vector<double>(k, 0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < k; ++j)
                for (int t = 0; t < m; ++t)
                    c[i][j] += a[i][t] * b[t][j];
        return c;
    }

    Fit BreslowEstimatorFit(const std::vector<double>& pred, const std::vector<double>& unique_times,
                            const std::vector<int>& n_events, const std::vector<int>& n_at_risk) {
        int n = pred.size();
        std::vector<double>risk_score(n);
        std::vector<double>cum_risk_score(n);
        double value = 0;
        for (int i = 0; i < n; ++i) {
            risk_score[i] = exp(pred[n - i - 1]);
            value += risk_score[i];
            cum_risk_score[i] = risk_score[i];
            if (i > 0)
                cum_risk_score[i] += cum_risk_score[i - 1];
        }
        int k = 0;
        std::vector<double>cum_baseline_hazard(n_at_risk.size());
        cum_baseline_hazard[0] = n_events[0] / value;
        for (int i = 1; i < n_at_risk.size(); ++i) {
            double d = n_at_risk[i - 1] - n_at_risk[i];
            value -= (cum_risk_score[k + d - 1] - (k > 0 ? cum_risk_score[k - 1] : 0));
            k += d;
            cum_baseline_hazard[i] = n_events[i] / value;
        }
        assert(k == n_at_risk[0] - n_at_risk.back());

        for (int i = 1; i < cum_baseline_hazard.size(); ++i)
            cum_baseline_hazard[i] += cum_baseline_hazard[i - 1];

        std::vector<double>baseline_survival;
        for (int i = 0; i < cum_baseline_hazard.size(); ++i)
            baseline_survival.push_back(exp(-cum_baseline_hazard[i]));

        return Fit(StepFunction(unique_times, baseline_survival));
    }

    void compute_counts(const std::vector<int>& event, const std::vector<double>& time,
                        std::vector<int> &unique_events, std::vector<double> &unique_times,
                        std::vector<int>& n_at_risk) {
        n_at_risk.push_back(0);
        int n_samples = event.size();
        double last_time = -1;
        for (int i = n_samples - 1; i >= 0; --i) {
            if (time[i] != last_time) {
                last_time = time[i];
                unique_times.push_back(last_time);
                unique_events.push_back(0);
                n_at_risk.push_back(0);
            }
            n_at_risk.back()++;
            if (event[i] == 1)
                unique_events.back()++;
        }
        for (int i = 1; i < n_at_risk.size(); ++i)
            n_at_risk[i] += n_at_risk[i - 1];
        for (int& count : n_at_risk)
            count = n_samples - count;
        n_at_risk.pop_back();
    }

    void prefit(std::vector<std::vector<double>>& x, std::vector<double>& x_offset,
                std::vector<double>& x_scale) {
        int nrows = x.size();
        int ncols = x[0].size();
        x_offset = std::vector<double>(ncols, 0);
        x_scale = std::vector<double>(ncols, 0);
        for (int i = 0; i < nrows; ++i) {
            for (int j = 0; j < ncols; ++j) {
                x_offset[j] += x[i][j];
            }
        }
        for (int j = 0; j < ncols; ++j) {
            x_offset[j] /= nrows;
        }
        for (int i = 0; i < nrows; ++i)
            for (int j = 0; j < ncols; ++j)
                x[i][j] -= x_offset[j];


        for (int i = 0; i < nrows; ++i) {
            for (int j = 0; j < ncols; ++j) {
                x_scale[j] += x[i][j] * x[i][j];
            }
        }
        for (int j = 0; j < ncols; ++j) {
            x_scale[j] = sqrt(x_scale[j]);
            if (x_scale[j] == 0)
                x_scale[j] = 1;
        }
        for (int i = 0; i < nrows; ++i)
            for (int j = 0; j < ncols; ++j)
                x[i][j] /= x_scale[j];
    }
    void update(std::vector<int>& aib, int pos, int val) {
        for (; pos < aib.size(); pos += (pos & -pos))
            aib[pos] += val;
    }

    int query(std::vector<int>& aib, int pos) {
        int ans = 0;
        for (; pos > 0; pos -= (pos & -pos))
            ans += aib[pos];
        return ans;
    }

    double CIndex(std::vector<double>pred, std::vector<double>time, std::vector<int>event) {
        std::vector<std::pair<double, int> >o;
        for (int  i = 0; i < pred.size(); ++i) {
            o.push_back({pred[i], i});
        }
        std::sort(o.begin(), o.end());
        std::vector<int>time_o(pred.size());
        double eps = 1e-6;
        int val = 0;
        for (int i = 0; i < o.size(); ++i) {
            int j = i;
            val++;
            while (j < o.size() && fabs(o[j].first - o[i].first) <= eps) {
                time_o[o[j].second] = val;
                j++;
            }
            i = j - 1;
        }
        std::vector<int>aib(val + 1, 0);
        double cc = 0, dc = 0, tr = 0;
        int total = 0;
        for (int i = 0; i < time.size(); ++i) {
            int j = i;
            while (j < time.size() && fabs(time[j] - time[i]) <= eps) {
                if (event[j]) {
                    int cc1 = query(aib, time_o[j] - 1);
                    int tr1 = query(aib, time_o[j]) - cc1;
                    int dc1 = total - cc1 - tr1;
                    cc += cc1;
                    dc += dc1;
                    tr += tr1;
                }
                j++;
            }
            total += j - i;
            j = i;
            while (j < time.size() && fabs(time[j] - time[i]) <= eps) {
                update(aib, time_o[j], 1);
                j++;
            }
            i = j - 1;
        }
        if (cc + tr + dc == 0)
            return 0;
        return (cc + 0.5 * tr) / (cc + tr + dc);
    }

    double getLeafCosts(const std::vector<double>& pred, const std::vector<double>& time, const std::vector<int>& event) {
        double part_sum = 0, log_like = 0;
        int i = 0;
        while (i < pred.size()) {
            double t = time[i];
            int j = i;
            while (i < pred.size() && time[i] == t) {
                part_sum += exp(pred[i]);
                i++;
            }
            while (j < i) {
                if (event[j])
                    log_like += pred[j] - log(part_sum);
                j++;
            }
        }

        return -log_like;
    }

    StepFunction predict_survival_function(const Fit& model, std::vector<double>x, double tol = 1e-4) {
        double prediction = 0;
        for (int i = 0; i < x.size(); ++i)
            prediction += x[i] * model.coefs[i];
        prediction -= model.offset;
        std::vector<double>y;
        for (int i = 0; i < model.func.y.size(); ++i)
            y.push_back(pow(model.func.y[i], exp(prediction)));

        return StepFunction(model.func.x, y);
    }

    Fit fit(const ADataView& data, double l1_ratio, std::string validation, int nalphas = 15, int n_iter = 100) {
        std::vector<double>l1_values;
        if (l1_ratio == -1) {
            l1_values = {0.2, 0.4, 0.6, 0.8};
        } else {
            l1_values = {l1_ratio};
        }
        Fit model;
        double best_score = DBL_MAX;

        auto rng = std::default_random_engine(0);
        std::vector<int> rnd;
        for (int i = 0; i < data.GetInstancesForLabel(0).size(); ++i)
            rnd.push_back(i);
        std::shuffle(rnd.begin(), rnd.end(), rng);
        int prop = (int) (0.2 * rnd.size());
        std::vector<int> test;
        for (int i = 0; i < prop; ++i)
            test.push_back(rnd[i]);

        std::sort(test.begin(), test.end());

        std::vector<std::vector<double>> x;
        std::vector<std::vector<double>> x_c;
        std::vector<std::vector<double>> x_test;
        int nrows_test = test.size();
        int nrows = data.GetInstancesForLabel(0).size() - nrows_test;
        int ncols = static_cast<const Instance<double, CSAData> *>(data.GetInstancesForLabel(
                0)[0])->GetExtraData().GetVars().size();
        std::vector<double> times;
        times.reserve(nrows);
        std::vector<int> events;
        events.reserve(nrows);
        std::vector<double> times_test;
        times.reserve(nrows_test);
        std::vector<int> events_test;
        events.reserve(nrows_test);
        int j = 0;
        for (int i = 0; i < data.GetInstancesForLabel(0).size(); ++i) {
            auto instance = static_cast<const Instance<double, CSAData> *>(data.GetInstancesForLabel(0)[i]);
            double time = instance->GetLabel();
            int event = instance->GetExtraData().GetEvent();
            std::vector<double> vars = instance->GetExtraData().GetVars();
            if (j < nrows_test && test[j] == i) { // add to validation set
                x_test.push_back(vars);
                times_test.push_back(time);
                events_test.push_back(event);
                j++;
            }
            else {
                x.push_back(vars);
                x_c.push_back(vars);
                times.push_back(time);
                events.push_back(event);
            }
        }
        std::vector<double> x_offset;
        std::vector<double> x_scale;
        prefit(x, x_offset, x_scale);

        for (double l1 : l1_values) {
            std::vector<double> flat_x;
            flat_x.reserve(nrows * ncols);
            for (int j = 0; j < ncols; ++j)
                for (int i = 0; i < nrows; ++i)
                    flat_x.push_back(x[i][j]);

            Eigen::Map<Eigen::MatrixXd> x_map(flat_x.data(), nrows, ncols);

            Eigen::Map<Eigen::VectorXd> time_map(times.data(), nrows);
            Eigen::Map<Eigen::VectorXuint8> event_map(events.data(), nrows);
            std::vector<double> pen;
            pen.reserve(ncols);
            for (int i = 0; i < ncols; ++i)
                pen.push_back(1);
            Eigen::Map<Eigen::VectorXd> pen_map(pen.data(), ncols);
            std::vector<double> alphas;
            alphas.reserve(nalphas);
            for (int i = 0; i < nalphas; ++i)
                alphas.push_back(0);
            Eigen::Map<Eigen::VectorXd> alphas_map(alphas.data(), nalphas);
            bool create_path = true;
            Eigen::MatrixXd::Scalar alpha_min_ratio = (nrows > ncols ? 0.0001 : 0.01);
            size_t max_iter = n_iter;
            double eps = 1e-6;
            bool verbose = false;
            std::vector<double> coef_path;
            coef_path.reserve(nalphas * ncols);
            for (int i = 0; i < nalphas * ncols; ++i)
                coef_path.push_back(0);
            Eigen::Map<Eigen::MatrixXd> coef_path_map(coef_path.data(), ncols, nalphas);
            std::vector<double> final_alphas;
            final_alphas.reserve(nalphas);
            for (int i = 0; i < nalphas; ++i)
                final_alphas.push_back(0);
            Eigen::Map<Eigen::VectorXd> final_alphas_map(final_alphas.data(), nalphas);
            std::vector<double> final_dev_ratio;
            final_dev_ratio.reserve(nalphas);
            for (int i = 0; i < nalphas; ++i)
                final_dev_ratio.push_back(0);

            Eigen::Map<Eigen::VectorXd> final_dev_ratio_map(final_dev_ratio.data(), nalphas);
            Eigen::MatrixXd::Scalar l1_ratio = l1;

            nalphas = fit_coxnet<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXuint8>
                    (x_map, time_map, event_map, pen_map, alphas_map, coef_path_map, final_alphas_map,
                     final_dev_ratio_map, create_path, alpha_min_ratio, l1_ratio, max_iter, eps, verbose);

            std::vector<std::vector<double> > coef(ncols, std::vector<double>(nalphas));
            for (int j = 0; j < nalphas; ++j) {
                for (int i = 0; i < ncols; ++i)
                    coef[i][j] = coef_path[j * ncols + i] / x_scale[i];
            }

            std::vector<std::vector<double> > predictions = dotProduct(x_c, coef);
            std::vector<std::vector<double> > offset = dotProduct(std::vector<std::vector<double>>(1, x_offset), coef);

            std::vector<double> unique_times;
            std::vector<int> n_events;
            std::vector<int> n_at_risk;
            compute_counts(events, times, n_events, unique_times, n_at_risk);

            std::vector<Fit> models;

            for (int i = 0; i < nalphas; ++i) {
                std::vector<double> pred_col(nrows);
                for (int j = 0; j < nrows; ++j)
                    pred_col[j] = predictions[j][i] - offset[0][i];
                models.push_back(BreslowEstimatorFit(pred_col, unique_times, n_events, n_at_risk));
                models.back().offset = offset[0][i];
                models.back().alpha = final_alphas[i];
                models.back().coefs = std::vector<double>(ncols);
                for (int j = 0; j < ncols; ++j)
                    models.back().coefs[j] = coef[j][i];

            }

            std::vector<std::vector<double> > predictions_test = dotProduct(x_test, coef);

            for (int i = 0; i < nalphas; ++i) {
                std::vector<double> pred_col_test(nrows_test);
                for (int j = 0; j < nrows_test; ++j)
                    pred_col_test[j] = predictions_test[j][i] - models[i].offset;
                double score = 0;
                if (validation == "log-like")
                    score = getLeafCosts(pred_col_test, times_test, events_test);
                else
                    score = -CIndex(pred_col_test, times_test, events_test);
                if (score < best_score) {
                    best_score = score;
                    model = models[i];
                }
            }
        }
        return model;
    }

    // Solve a leaf node by finding the minimum negative log-likelihood estimate for beta with its corresponding loss
    Node <CoxSurvivalAnalysis>
    CoxSurvivalAnalysis::SolveLeafNode(const ADataView &data, const ContextType &context) const {
        int ncols = 0, nrows = data.GetInstancesForLabel(0).size(), nevents = 0;
        for (int i = 0; i < nrows; ++i) {
            auto instance = static_cast<const Instance<double, CSAData> *>(data.GetInstancesForLabel(0)[i]);
            int event = instance->GetExtraData().GetEvent();
            std::vector<double> vars = instance->GetExtraData().GetVars();
            ncols = vars.size();
            nevents += event;
        }
        if (nevents < 10 * ncols) {
            return Node<CoxSurvivalAnalysis>(Fit(), DBL_MAX);
        }
        Fit model = fit(data, l1_ratio, validation_technique);
        return Node<CoxSurvivalAnalysis>(model, GetLeafCosts(data, context, model));
    }

    // Get the loss for a leaf node given a beta estimate
    double CoxSurvivalAnalysis::GetLeafCosts(const ADataView &data, const ContextType &context, const Fit& model) const {
        std::vector<double> pred;
        std::vector<double> time;
        std::vector<int> event;
        for (const auto i: data.GetInstancesForLabel(0)) {
            auto instance = static_cast<const Instance<double, CSAData>*>(i);
            std::vector<double> vars = instance->GetExtraData().GetVars();
            double s = 0;
            for (int j = 0; j < vars.size(); ++j) {
                s += vars[j] * model.coefs[j];
            }
            pred.push_back(s);
            time.push_back(instance->GetLabel());
            event.push_back(instance->GetExtraData().GetEvent());
        }
        return getLeafCosts(pred, time, event);
    }

    TuneRunConfiguration
    CoxSurvivalAnalysis::GetTuneRunConfiguration(const ParameterHandler &default_config, const ADataView &data,
                                                 int phase) {
        TuneRunConfiguration config;

        int max_nodes = int(default_config.GetIntegerParameter("max-num-nodes"));
        int max_d = int(default_config.GetIntegerParameter("max-depth"));

        for (int d = max_d; d <= max_d; d++) {
            int _max_nodes = std::min(max_nodes, (1 << d) - 1);
            for (int i = std::max(0, _max_nodes - 1); i <= _max_nodes; i++) {
                ParameterHandler params = default_config;
                params.SetIntegerParameter("max-depth", d);
                params.SetIntegerParameter("max-num-nodes", i);
                config.AddConfiguration(params, "d=" + std::to_string(d) + ", n=" + std::to_string(i));
            }
        }
        config.runs = 1;
        config.reset_solver = true;
        return config;
    }

    double CoxSurvivalAnalysis::Classify(const AInstance* instance, const Fit& label) const {
        std::vector<double> vars = static_cast<const Instance<double, CSAData>*>(instance)->GetExtraData().GetVars();
        StepFunction survival_function = predict_survival_function(label, vars);
        double area = 0;
        for (int i = 0; i < survival_function.x.size(); ++i) {
            if (i == 0)
                area += survival_function.x[i];
            else
                area += survival_function.y[i - 1] * (survival_function.x[i] - survival_function.x[i - 1]);
        }
        double half_area = area / 2.0;
        area = 0;
        for (int i = 0; i < survival_function.x.size(); ++i) {
            double prev_area = area;
            if (i == 0)
                area += survival_function.x[i];
            else
                area += survival_function.y[i - 1] * (survival_function.x[i] - survival_function.x[i - 1]);
            if (area >= half_area) {
                double rem_area = half_area - prev_area;
                if (i == 0)
                    return rem_area;
                return rem_area / survival_function.y[i - 1] + survival_function.x[i - 1];
            }
        }
    }
}
