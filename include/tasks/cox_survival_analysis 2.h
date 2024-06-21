#pragma once
#include "tasks/optimization_task.h"
#include <functional>
#include <vector>
#include "tasks/cox_regression/fit-object.h"

namespace STreeD {

    // Extra data for Survival analysis
    class CSAData {
    public:
        static CSAData ReadData(std::istringstream& iss, int num_labels);

        CSAData() = default;

        inline CSAData(int ev, const std::vector<double>& vars) {
            this->ev = ev;
            this->vars = vars;
        };

        // Get the event (censoring = 0, or death = 1)
        inline int GetEvent() const { return ev; }

        // Get the coefficients values.
        inline std::vector<double> GetVars() const { return vars; }


    protected:
        int ev{ 0 };				// The event (censoring = 0, or death = 1)
        std::vector<double> vars;   // The realisation of the covariates.
    };

    // Not used
    struct CD2SASol {
        double hazard_sum{ 0 };
        int event_sum{ 0 };
        double negative_log_hazard_sum{ 0 };

        inline const CD2SASol& operator+=(const CD2SASol& v2) {
            hazard_sum += v2.hazard_sum;
            event_sum += v2.event_sum;
            negative_log_hazard_sum += v2.negative_log_hazard_sum;
            return *this;
        }
        inline CD2SASol operator+(const CD2SASol& v2) const { return CD2SASol(*this) += v2; }
        inline const CD2SASol& operator-=(const CD2SASol& v2) {
            hazard_sum -= v2.hazard_sum;
            event_sum -= v2.event_sum;
            negative_log_hazard_sum -= v2.negative_log_hazard_sum;
            return *this;
        }
        inline CD2SASol operator-(const CD2SASol& v2) const { return CD2SASol(*this) -= v2; }
        inline bool operator==(const CD2SASol& v2) const {
            return std::abs(hazard_sum - v2.hazard_sum) < 1e-6
                   && event_sum == v2.event_sum
                   && std::abs(negative_log_hazard_sum - v2.negative_log_hazard_sum) < 1e-6;
        }
        inline bool operator!=(const CD2SASol& v2) const { return !(*this == v2); }
    };


    class CoxSurvivalAnalysis : public OptimizationTask {
    private:
        // extra data instances created in preprocessing
        AData train_data_storage, test_data_storage;
        double l1_ratio = 0.99;
        std::string validation_technique = "log-like";

    public:
        using ET = CSAData;				            	// The extra data type
        using SolType = double;			               	// The type of the loss is double
        using SolLabelType = Fit;		                // The type of the estimate is 'Fit'
        using SolD2Type = CD2SASol;			            // The type of the depth-two solution - n/a
        using TestSolType = double;			            // The type of the test loss is double
        using LabelType = double;		            	// The type of the label in the data set (time of event)

        static const bool preprocess_train_test_data = false;	// No preprocessing
        static const bool use_terminal = false;					// deactivates the depth-two solver
        static const bool element_additive = false;				// deactivates the similarity lower bound

        static const bool total_order = true;			// This optimization task is totally ordered
        static const bool custom_leaf = true;			// A custom leaf node optimization function is provided
        static constexpr  double worst = DBL_MAX;		// The worst solution value (infinite loss)
        static constexpr  double best = 0;				// The best solution value (zero loss)
        static const Fit worst_label;

        CoxSurvivalAnalysis(const ParameterHandler& parameters) {
            l1_ratio = parameters.GetFloatParameter("l1-ratio");
            validation_technique = parameters.GetStringParameter("survival-validation");
        }

        inline void UpdateParameters(const ParameterHandler& parameters) {
            l1_ratio = parameters.GetFloatParameter("l1-ratio");
            validation_technique = parameters.GetStringParameter("survival-validation");
        }

        // Solve a leaf node by finding the max-likelihood estimate for the coefficients with its corresponding loss
        Node<CoxSurvivalAnalysis> SolveLeafNode(const ADataView& data, const ContextType& context) const;

        // Get the loss for a leaf node given a beta estimate
        double GetLeafCosts(const ADataView& data, const ContextType& context, const Fit& beta) const;

        // Get the test loss for a leaf node given a beta estimate
        inline double GetTestLeafCosts(const ADataView& data, const ContextType& context, const Fit& beta) const { return GetLeafCosts(data, context, beta);}

        // Compute the training score (average loss)
        inline double ComputeTrainScore(double train_value) const { return train_value / train_summary.size; }

        // Compute the test score on the training data (average loss)
        inline double ComputeTrainTestScore(double train_value) const { return train_value / train_summary.size; }

        // Compute the test score on the test data (average loss)
        inline double ComputeTestTestScore(double test_value) const { return test_value / test_summary.size; }

        // Compare two scores (lower is better)
        inline static bool CompareScore(double score1, double score2) { return score1 < score2; } // return true if score1 is better than score2

        // Classify an instance, return the theta estimate
        double Classify(const AInstance* instance, const Fit& label) const ;

        inline static std::string SolToString(const Fit& label) { return label.ToString(); }

        // Get the depth two costs for the given instance (event sum, hazard sum, negative log hazard sum)
        void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, SolD2Type& costs, int multiplier) {  }

        // Compute the loss from the depth-two cost tuple
        void ComputeD2Costs(const SolD2Type& d2costs, int count, double& costs) {}

        // Return true if the depth-two contribution is zero (always false)
        inline bool IsD2ZeroCost(const SolD2Type& d2costs) const { return false; }

        // Compute the max-likelihood theta estimate from the depht-two cost tuple
        Fit GetLabel(const SolD2Type& costs, int count) {return Fit(); }

        // Get the configurations for hypertuning
        static TuneRunConfiguration GetTuneRunConfiguration(const ParameterHandler& default_config, const ADataView& train_data, int phase);
    };

}