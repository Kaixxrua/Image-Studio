package client

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"
)

func TestBuildGeminiPayloadUsesGenerateContentShape(t *testing.T) {
	raw, err := BuildGeminiPayload(Options{
		Prompt:       "画一只发光的猫",
		Size:         "2048x1152",
		OutputFormat: "webp",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v := mustDecodePayload(t, raw)

	contents := v["contents"].([]any)
	if len(contents) != 1 {
		t.Fatalf("contents len = %d, want 1", len(contents))
	}
	message := contents[0].(map[string]any)
	if message["role"] != "user" {
		t.Fatalf("role = %v, want user", message["role"])
	}
	parts := message["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("parts len = %d, want 1", len(parts))
	}
	textPart := parts[0].(map[string]any)
	if textPart["text"] != "画一只发光的猫" {
		t.Fatalf("text = %v", textPart["text"])
	}

	generationConfig := v["generationConfig"].(map[string]any)
	modalities := generationConfig["responseModalities"].([]any)
	if len(modalities) != 2 || modalities[0] != "TEXT" || modalities[1] != "IMAGE" {
		t.Fatalf("responseModalities = %v", modalities)
	}
	responseFormat := generationConfig["responseFormat"].(map[string]any)
	image := responseFormat["image"].(map[string]any)
	if image["aspectRatio"] != "16:9" {
		t.Fatalf("aspectRatio = %v, want 16:9", image["aspectRatio"])
	}
	if image["imageSize"] != "2K" {
		t.Fatalf("imageSize = %v, want 2K", image["imageSize"])
	}
	if image["mimeType"] != "image/webp" {
		t.Fatalf("mimeType = %v, want image/webp", image["mimeType"])
	}
}

func TestBuildGeminiPayloadIncludesInlineDataFromDataURL(t *testing.T) {
	b64 := base64.StdEncoding.EncodeToString([]byte("reference"))
	raw, err := BuildGeminiPayload(Options{
		Prompt:        "参考这张图",
		ImageDataURLs: []string{"data:image/jpeg;base64," + b64},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v := mustDecodePayload(t, raw)
	parts := v["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts len = %d, want 2", len(parts))
	}
	inlineData := parts[1].(map[string]any)["inline_data"].(map[string]any)
	if inlineData["mime_type"] != "image/jpeg" {
		t.Fatalf("mime_type = %v, want image/jpeg", inlineData["mime_type"])
	}
	if inlineData["data"] != b64 {
		t.Fatalf("data = %v, want %s", inlineData["data"], b64)
	}
}

func TestBuildGeminiPayloadReadsLocalImagePath(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "source.png")
	if err := os.WriteFile(src, fakePNG, 0o644); err != nil {
		t.Fatal(err)
	}
	dataURLB64 := base64.StdEncoding.EncodeToString([]byte("second-reference"))

	raw, err := BuildGeminiPayload(Options{
		Prompt:        "参考本地图",
		ImagePaths:    []string{src},
		ImageDataURLs: []string{"data:image/webp;base64," + dataURLB64},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v := mustDecodePayload(t, raw)
	parts := v["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if len(parts) != 3 {
		t.Fatalf("parts len = %d, want 3", len(parts))
	}
	inlineData := parts[1].(map[string]any)["inline_data"].(map[string]any)
	if inlineData["mime_type"] != "image/png" {
		t.Fatalf("mime_type = %v, want image/png", inlineData["mime_type"])
	}
	if inlineData["data"] != base64.StdEncoding.EncodeToString(fakePNG) {
		t.Fatalf("unexpected inline data")
	}
	secondInlineData := parts[2].(map[string]any)["inline_data"].(map[string]any)
	if secondInlineData["mime_type"] != "image/webp" || secondInlineData["data"] != dataURLB64 {
		t.Fatalf("second inline_data = %v", secondInlineData)
	}
}

func TestBuildGeminiPayloadRejectsTooManyInputImages(t *testing.T) {
	urls := make([]string, 0, maxGeminiInputImages+1)
	b64 := base64.StdEncoding.EncodeToString([]byte("reference"))
	for i := 0; i < maxGeminiInputImages+1; i++ {
		urls = append(urls, "data:image/png;base64,"+b64)
	}
	if _, err := BuildGeminiPayload(Options{Prompt: "太多参考图", ImageDataURLs: urls}); err == nil {
		t.Fatal("expected error for too many Gemini input images")
	}
}

func TestBuildImagenPayloadUsesPredictShape(t *testing.T) {
	raw, err := BuildImagenPayload(Options{
		Prompt:            "未来城市海报",
		Size:              "1152x2048",
		OutputFormat:      "jpeg",
		ImagenSampleCount: 2,
		NegativePrompt:    "低清晰度",
		Seed:              123,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v := mustDecodePayload(t, raw)
	instances := v["instances"].([]any)
	if len(instances) != 1 {
		t.Fatalf("instances len = %d, want 1", len(instances))
	}
	if instances[0].(map[string]any)["prompt"] != "未来城市海报" {
		t.Fatalf("prompt = %v", instances[0].(map[string]any)["prompt"])
	}
	parameters := v["parameters"].(map[string]any)
	if parameters["sampleCount"] != float64(2) {
		t.Fatalf("sampleCount = %v, want 2", parameters["sampleCount"])
	}
	if parameters["aspectRatio"] != "9:16" {
		t.Fatalf("aspectRatio = %v, want 9:16", parameters["aspectRatio"])
	}
	if parameters["imageSize"] != "2K" {
		t.Fatalf("imageSize = %v, want 2K", parameters["imageSize"])
	}
	if parameters["negativePrompt"] != "低清晰度" {
		t.Fatalf("negativePrompt = %v", parameters["negativePrompt"])
	}
	if parameters["seed"] != float64(123) {
		t.Fatalf("seed = %v, want 123", parameters["seed"])
	}
	outputOptions := parameters["outputOptions"].(map[string]any)
	if outputOptions["mimeType"] != "image/jpeg" {
		t.Fatalf("mimeType = %v, want image/jpeg", outputOptions["mimeType"])
	}
}

func TestGeminiAndImagenExplicitOptionsOverrideDerivedConfig(t *testing.T) {
	geminiAspect, geminiSize := effectiveGeminiImageConfig(Options{
		Size:              "2048x1152",
		GeminiAspectRatio: "3:4",
		GeminiImageSize:   "4k",
	})
	if geminiAspect != "3:4" || geminiSize != "4K" {
		t.Fatalf("gemini config = %s/%s, want 3:4/4K", geminiAspect, geminiSize)
	}

	imagenAspect, imagenSize := effectiveImagenImageConfig(Options{
		Size:              "3840x2160",
		ImagenAspectRatio: "4:3",
		ImagenImageSize:   "2k",
	})
	if imagenAspect != "4:3" || imagenSize != "2K" {
		t.Fatalf("imagen config = %s/%s, want 4:3/2K", imagenAspect, imagenSize)
	}
}

func TestBuildGeminiAndImagenPayloadRejectEmptyPrompt(t *testing.T) {
	if _, err := BuildGeminiPayload(Options{Prompt: "  "}); err != ErrEmptyPrompt {
		t.Fatalf("BuildGeminiPayload error = %v, want ErrEmptyPrompt", err)
	}
	if _, err := BuildImagenPayload(Options{Prompt: "  "}); err != ErrEmptyPrompt {
		t.Fatalf("BuildImagenPayload error = %v, want ErrEmptyPrompt", err)
	}
}

func TestEffectiveGeminiAndImagenModelsUseProviderDefaults(t *testing.T) {
	if got := EffectiveGeminiImageModel(Options{}); got != DefaultGeminiImageModel {
		t.Fatalf("gemini model = %s, want %s", got, DefaultGeminiImageModel)
	}
	if got := EffectiveImagenModel(Options{}); got != DefaultImagenModel {
		t.Fatalf("imagen model = %s, want %s", got, DefaultImagenModel)
	}
	if got := EffectiveGeminiImageModel(Options{ImageModelID: " custom-gemini "}); got != "custom-gemini" {
		t.Fatalf("custom gemini model = %s", got)
	}
	if got := EffectiveImagenModel(Options{ImageModelID: " custom-imagen "}); got != "custom-imagen" {
		t.Fatalf("custom imagen model = %s", got)
	}
}
