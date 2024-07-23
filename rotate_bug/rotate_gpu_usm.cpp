#include <CL/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <complex>
#include <mkl.h>
using namespace sycl;

int main() {
//    double c = 0.5550150321048569;
//    std::complex<double> s(0.83184031769183,0.0); 
//      double c = 0.40516847542537016;
//      std::complex<double> s(0.9142420393536284,0.0);

	// Occasional failure
      double c = 0.8187587885612081;
      std::complex<double> s(0.5741376543598085,0.0);
    std::cout << "c: " << c << "s: " << s << std::endl; 
    try {
        // Create a SYCL queue
    cl::sycl::queue main_queue(cl::sycl::gpu_selector{});
	auto cxt = main_queue.get_context();
	auto dev = main_queue.get_device();
#if 0
	auto ua = usm_allocator<std::complex<double>, usm::alloc::shared, 64>(cxt, dev);
        std::vector<std::complex<double>, decltype(ua)> x(ua), y(ua);
	for (int i = 0; i < 1; i++) {
		x.push_back({1.0, 0.0});
		y.push_back({1.0, 0.0});
	}
#endif
	std::complex<double> *x = (std::complex<double> *) malloc_shared(10 * sizeof(std::complex<double>), dev, cxt); 
	std::complex<double> *y = (std::complex<double> *) malloc_shared(10 * sizeof(std::complex<double>), dev, cxt); 
       
	for (int i = 0; i < 1; i++) {
		x[i] = {1,0};
		y[i] = {1,0};
	}
	
	for (int i = 0; i < 1; i++) {
		std::cout << x[i] << std::endl;
	}

	// Perform the Givens rotation on the vectors
        auto status = oneapi::mkl::blas::column_major::rot(main_queue, 1, x, 1, 
							y, 1, c, s);
	sycl::get_native<sycl::backend::ext_oneapi_level_zero>(status);
	main_queue.wait_and_throw();
	std::cout << "GPU Results: "<< std::endl; 
	    int n = 1;
	    std::cout << n << std::endl;
	    std::cout << "X - array\n";
	    for(int i = 0; i < n; i++) {
		std::cout << x[i] << " ";
	    }
	    std::cout << std::endl;

	    std::cout << "Y - array\n";
	    for(int i = 0; i < n; i++) {
		std::cout << y[i] << " ";
	    }
	    std::cout << std::endl;


    } catch (cl::sycl::exception& e) {
        std::cout << "SYCL exception encountered: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Done with GPU Rotate !!, Starting CPU rotate" << std::endl;

    std::vector<std::complex<double> > x;
    std::vector<std::complex<double> >y;
    for (int i = 0; i < 1; i++) {
	x.push_back({1.0,0.0});
	y.push_back({1.0,0.0});
    }
    int n = sizeof(x) / sizeof(x[0]);

    cblas_zrot(n, x.data(), 1, y.data(), 1, c, &s);
    std::cout << "CPU Results: "<< std::endl; 

    std::cout << "X - array\n";
    for(int i = 0; i < n; i++) {
    	std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Y - array\n";
    for(int i = 0; i < n; i++) {
    	std::cout << y[i] << " ";
    }
    std::cout << std::endl;


    return 0;
}

