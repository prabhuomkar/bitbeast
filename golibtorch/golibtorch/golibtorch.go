package golibtorch

// #cgo LDFLAGS: -lstdc++ -L/usr/local/libtorch/lib -ltorch_cpu -lc10
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #include <stdio.h>
// #include <stdlib.h>
// #include "golibtorch.h"
import "C"
import (
	"image"
	"unsafe"

	"errors"

	"github.com/disintegration/imaging"
)

const (
	NumChannels = 3
	NumLabels   = 1000 // imagenet labels
	BatchSize   = 1
	TopK        = 5
	ImageSize   = 224
)

type Model struct {
	model C.mModel
}

func NewModel(modelFile string) (*Model, error) {
	return &Model{
		model: C.NewModel(
			C.CString(modelFile),
		),
	}, nil
}

func (m *Model) GetResult(img image.Image) (map[string]float32, error) {
	// prepare input
	data, vals, err := preProcess(
		img, []float32{0.485, 0.456, 0.406}, []float32{0.229, 0.224, 0.225})
	if err != nil {
		return nil, err
	}

	// get result from CGO
	inputPtr := (*C.float)(unsafe.Pointer(&data[0]))
	channelPtr := (*C.int)(unsafe.Pointer(&vals[0]))
	widthPtr := (*C.int)(unsafe.Pointer(&vals[1]))
	heightPtr := (*C.int)(unsafe.Pointer(&vals[2]))
	cResult := C.GetResult(m.model, inputPtr, channelPtr, widthPtr, heightPtr)
	defer C.free(unsafe.Pointer(cResult))
	if cResult == nil {
		return nil, errors.New("error in getting result")
	}

	labels := make([]string, TopK)
	scores := make([]float32, TopK)
	scoresHeader := (*[1 << 30]float32)(unsafe.Pointer(cResult.scores))[:TopK:TopK]
	copy(scores, scoresHeader)
	ptr := uintptr(unsafe.Pointer(cResult.labels))
	for i := 0; i < TopK; i++ {
		labelPtr := (**C.char)(unsafe.Pointer(ptr))
		labels[i] = C.GoString(*labelPtr)
		ptr += unsafe.Sizeof(uintptr(0))
	}

	// prepare output
	result := make(map[string]float32)
	for i := 0; i < TopK; i++ {
		result[labels[i]] = scores[i]
	}

	return result, nil
}

func (m *Model) DeleteModel() {
	C.DeleteModel(m.model)
}

// preprocessing image (not exactly like pytorch transforms but gets the job done)
func preProcess(img image.Image, mean []float32, stddev []float32) ([]float32, []int, error) {
	if img == nil {
		return nil, nil, errors.New("error due to invalid image")
	}
	img = imaging.Fill(img, ImageSize, ImageSize, imaging.Center, imaging.Lanczos)
	bounds := img.Bounds()
	w := bounds.Max.X - bounds.Min.X
	h := bounds.Max.Y - bounds.Min.Y
	data := make([]float32, NumChannels*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			data[y*w+x] = ((float32(r>>8) / 255.0) - mean[0]) / stddev[0]
			data[w*h+y*w+x] = ((float32(g>>8) / 255.0) - mean[1]) / stddev[1]
			data[2*w*h+y*w+x] = ((float32(b>>8) / 255.0) - mean[2]) / stddev[2]
		}
	}
	return data, []int{NumChannels, w, h}, nil
}
