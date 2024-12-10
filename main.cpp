#include <iostream>
#include <Eigen/Dense> //This line includes the Eigen library for matrix calulations
#include <vector>// Vector library

//This line makes sure you can use the Eigen syntax language
using namespace Eigen;
using namespace std;

// Function to calculate the Predicted State Matrix
MatrixXd getPredStateMatrix(double deltaT, double acceleration, double measuredPosition, double measuredVelocity,
                            const MatrixXd& A, const MatrixXd& errorMatrix) {
    // Previous state matrix (X_k-1)
    MatrixXd previousStateMatrix(2, 1);
    previousStateMatrix << measuredPosition, measuredVelocity;

    // Control matrix (B)
    MatrixXd B(2, 1);
    B << 0.5 * deltaT * deltaT,
         deltaT;

    // Control variable matrix (mu_k)
    MatrixXd controlVariableMatrix(1, 1);
    controlVariableMatrix << acceleration;

    // A * X_k-1 + B * mu_k
    MatrixXd predictedStateMatrix = A * previousStateMatrix + B * controlVariableMatrix;

    //w_k
    if (errorMatrix.rows() > 0 && errorMatrix.cols() > 0) {
        predictedStateMatrix += errorMatrix;
    }

    return predictedStateMatrix;
}

// Function to initialize the Process Covariance Matrix
MatrixXd getInitProcessCVMatrix(double processPositionError, double processVelocityError) {

    // Process covariance matrix (P_k-1)
    MatrixXd processCovarianceMatrix(2, 2);
    processCovarianceMatrix << pow(processPositionError, 2), 0,
                               0, pow(processVelocityError, 2);

    return processCovarianceMatrix;
}

// Function to calculate the Predicted Process Covariance Matrix
MatrixXd getPredProcessCVMatrix(const MatrixXd& initProcessCVMatrix, const MatrixXd& A, const MatrixXd& Q) {

    // A * P_k-1 * A^T
    MatrixXd firstTerm = A * initProcessCVMatrix * A.transpose();

    // Add process noise (Q) if provided
    if (Q.rows() > 0 && Q.cols() > 0) {
        return firstTerm + Q;
    }

    return firstTerm;
}

// Function to calculate the Kalman Gain
MatrixXd getKalmanGain(const MatrixXd& predProcessCVMatrix, double observationPositionError,
                       double observationVelocityError, const MatrixXd& H) {

    // Observation noise covariance matrix (R)
    MatrixXd R(2, 2);
    R << pow(observationPositionError, 2), 0,
         0, pow(observationVelocityError, 2);

    // K = P_kp * H^T * (H * P_kp * H^T + R)^-1
    MatrixXd S = H * predProcessCVMatrix * H.transpose() + R;

    return predProcessCVMatrix * H.transpose() * S.inverse();
}

// Function to get a New Observation
MatrixXd getNewObservation(double measuredPosition, double measuredVelocity, const MatrixXd& errorMatrix) {

    // Measured values matrix (Y_km)
    MatrixXd measuredValuesMatrix(2, 1);
    measuredValuesMatrix << measuredPosition, measuredVelocity;

    // Add error matrix if provided
    if (errorMatrix.rows() > 0 && errorMatrix.cols() > 0) {
        return measuredValuesMatrix + errorMatrix;
    }

    return measuredValuesMatrix;
}

// Function to calculate the Filtered State
MatrixXd calculateFilteredState(const MatrixXd& kalmanGain, const MatrixXd& predStateMatrix,
                                const MatrixXd& observationMatrix, const MatrixXd& H) {

    // X_k = X_kp + K * (Y_k - H * X_kp)
    MatrixXd predObsDifference = observationMatrix - H * predStateMatrix;

    return predStateMatrix + kalmanGain * predObsDifference;
}

// Function to update the Process Covariance Matrix
MatrixXd updateProcessCVMatrix(const MatrixXd& kalmanGain, const MatrixXd& H, const MatrixXd& predProcessCVMatrix) {

    // P_k = (I - K * H) * P_kp
    MatrixXd I = MatrixXd::Identity(2, 2);

    return (I - kalmanGain * H) * predProcessCVMatrix;
}

// Main Kinematics Kalman Filter Function
void kinematicsKalman(double deltaT, double acceleration, double measuredPosition, double measuredVelocity,
                      double processPositionError, double processVelocityError, double observationPositionError,
                      double observationVelocityError, double secondMeasuredPosition, double secondMeasuredVelocity,
                      vector<pair<double, double>> dataset, const MatrixXd& prevProcessCVMatrix) {

    // State transition matrix (A)
    MatrixXd A(2, 2);
    A << 1, deltaT,
         0, 1;

    // Observation matrix (H)
    MatrixXd H(2, 2);
    H << 1, 0,
         0, 1;

    // Predicted state matrix (X_kp)
    MatrixXd predictedStateMatrix = getPredStateMatrix(deltaT, acceleration, measuredPosition, measuredVelocity, A, MatrixXd());

    // Process covariance matrix (P_k-1)
    MatrixXd processCovarianceMatrix = prevProcessCVMatrix.rows() == 0
                                       ? getInitProcessCVMatrix(processPositionError, processVelocityError)
                                       : prevProcessCVMatrix;

    // Predicted process covariance matrix (P_kp)
    MatrixXd predictedProcessCovarianceMatrix = getPredProcessCVMatrix(processCovarianceMatrix, A, MatrixXd());

    // Kalman gain (K)
    MatrixXd kalmanGain = getKalmanGain(predictedProcessCovarianceMatrix, observationPositionError, observationVelocityError, H);

    // New observation (Y_k)
    MatrixXd newObservation = getNewObservation(secondMeasuredPosition, secondMeasuredVelocity, MatrixXd());

    // Filtered state (X_k)
    MatrixXd filteredState = calculateFilteredState(kalmanGain, predictedStateMatrix, newObservation, H);

    // Updated process covariance matrix (P_k)
    MatrixXd updatedProcessCovarianceMatrix = updateProcessCVMatrix(kalmanGain, H, predictedProcessCovarianceMatrix);

    // Output the results
    cout << "\n The predicted position is:\n " << filteredState(0, 0) << endl;
    cout << "\n The predicted velocity is:\n " << filteredState(1, 0) << endl;
    cout << "\n The new process covariance matrix is:\n " << updatedProcessCovarianceMatrix << endl;

    // Remove the first item from the dataset
    dataset.erase(dataset.begin());
    if (dataset.size() <= 1) {
        cout << "Kalman filter finished." << endl;
        return;
    }

    // Recursive call
    kinematicsKalman(deltaT, acceleration, filteredState(0, 0), filteredState(1, 0), processPositionError,
                     processVelocityError, observationPositionError, observationVelocityError, dataset[0].first,
                     dataset[0].second, dataset, updatedProcessCovarianceMatrix);
}

// Main Function
int main() {
    // Dataset: Position and velocity pairs
    vector<pair<double, double>> dataset = {
        {4000, 280}, {4260, 282}, {4550, 285}, {4860, 286}, {5110, 290}
    };

    // Initial call to the kinematics Kalman filter
    kinematicsKalman(1, 2, dataset[0].first, dataset[0].second, 20, 5, 25, 6, dataset[1].first, dataset[1].second, dataset, MatrixXd());
    return 0;
}
