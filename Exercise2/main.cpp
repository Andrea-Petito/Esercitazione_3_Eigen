#include <iostream>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

double rel_error(const VectorXd& x_computed, const VectorXd& x_true) {
	return (x_computed - x_true).norm() / x_true.norm();
}

int main() {
    VectorXd x_true(2);
	x_true << -1.0, -1.0;
	
	MatrixXd A1(2,2);
	A1 << 5.547001962252291e-01, -3.770900990025203e-02,	8.320502943378437e-01, -9.992887623566787e-01;
	
	VectorXd b1(2);
	b1 << -5.169911863249772e-01, 1.672384680188350e-01;
	
	MatrixXd A2(2,2);
	A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
	
	VectorXd b2(2);
	b2 << -6.394645785530173e-04, 4.259549612877223e-04;
	
	MatrixXd A3(2,2);
	A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;
	
	VectorXd b3(2);
	b3 << -6.400391328043042e-10, 4.266924591433963e-10;
	
	MatrixXd A[3] = {A1, A2, A3};
	VectorXd b[3] = {b1, b2, b3};
	
	for (int i = 0; i < 3; ++i) {
		cout << "Sistema" << i + 1 << ": " << endl;
		
		FullPivLU<MatrixXd> lu_decomp(A[i]);
        VectorXd x_lu = lu_decomp.solve(b[i]);
        double error_lu = rel_error(x_lu, x_true);
        cout << "  Soluzione LU: " << x_lu.transpose() << std::endl;
        cout << "  Errore relativo LU: " << error_lu << std::endl;
		
		HouseholderQR<MatrixXd> qr_decomp(A[i]);
        VectorXd x_qr = qr_decomp.solve(b[i]);
        double error_qr = rel_error(x_qr, x_true);
        cout << "  Soluzione QR: " << x_qr.transpose() << std::endl;
        cout << "  Errore relativo QR: " << error_qr << std::endl;
		
		cout << endl;
	}
	
    return 0;
}
