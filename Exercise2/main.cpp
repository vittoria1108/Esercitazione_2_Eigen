#include "Eigen/Eigen"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Eigen;


Vector2d SolveSystemPALU(const Matrix2d& A,
                         const Vector2d& b)
{
    Vector2d solutionPALU = A.fullPivLu().solve(b);     // PALU decomposition

    return solutionPALU;
}


Vector2d SolveSystemQR(const Matrix2d& A,
                       const Vector2d& b)
{
    Vector2d solutionQR = A.householderQr().solve(b);   // QR decomposition

    return solutionQR;
}


void ErrRel(const Matrix2d& A,
            const Vector2d& b,
            const Vector2d& exactSolution,
            double& errRelPALU,
            double& errRelQR)

{
    errRelPALU = (SolveSystemPALU(A,b) - exactSolution).norm() / exactSolution.norm();

    errRelQR = (SolveSystemQR(A,b) - exactSolution).norm() / exactSolution.norm();
}

int main()
{
    Vector2d exactSolution = {-1.0e+0, -1.0e+00};


    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,      // first row
        8.320502943378437e-01, -9.992887623566787e-01;      // second row

    Vector2d b1 = {-5.169911863249772e-01, 1.672384680188350e-01};

    double ErrRelPALU1;
    double ErrRelQR1;
    ErrRel(A1,b1,exactSolution,ErrRelPALU1,ErrRelQR1);

    cout << scientific <<setprecision(16) << "1: Relative error PALU: " << ErrRelPALU1 << " , " << "Relative Error QR: " << ErrRelQR1 << endl;


    Matrix2d A2;
    A2 << 5.547001962252291e-01,-5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;

    Vector2d b2 = {-6.394645785530173e-04, 4.259549612877223e-04};

    double ErrRelPALU2;
    double ErrRelQR2;
    ErrRel(A2,b2,exactSolution,ErrRelPALU2,ErrRelQR2);

    cout << scientific <<setprecision(16) << "2: Relative Error PALU: " << ErrRelPALU2 << " , " << "Relative Error QR: " << ErrRelQR2 << endl;


    Matrix2d A3;
    A3 << 5.547001962252291e-01,-5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b3 = {-6.400391328043042e-10, 4.266924591433963e-10};

    double ErrRelPALU3;
    double ErrRelQR3;
    ErrRel(A3,b3,exactSolution,ErrRelPALU3,ErrRelQR3);

    cout << scientific <<setprecision(16) << "3: Relative Error PALU: " << ErrRelPALU3 << " , " << "Relative Error QR: " << ErrRelQR3 << endl;

    return 0;
}
