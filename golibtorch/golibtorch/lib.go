package golibtorch

// #cgo LDFLAGS: -lstdc++ -L/usr/lib/libtorch/lib -ltorch_cpu -lc10
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
import "C"
