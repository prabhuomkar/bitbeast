package main

import (
	"bitbeast/golibtorch/golibtorch"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
)

func main() {
	model, err := golibtorch.NewModel("model/ResNet50_Quantized_IMAGENET1K_FBGEMM_V2.pt")
	if err != nil {
		log.Fatal(err)
	}
	defer model.DeleteModel()

	imgFile, err := os.Open("model/example.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer imgFile.Close()

	img, _, err := image.Decode(imgFile)
	if err != nil {
		log.Fatal(err)
	}

	result, err := model.GetResult(img)
	if err != nil {
		log.Fatal(err)
	}

	for label, score := range result {
		fmt.Printf("%s: %v\n", label, score)
	}
}
