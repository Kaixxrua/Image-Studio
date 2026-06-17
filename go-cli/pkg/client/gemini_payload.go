package client

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	defaultGeminiAspectRatio = "1:1"
	defaultGeminiImageSize   = "1K"
	defaultImagenAspectRatio = "1:1"
	defaultImagenImageSize   = "1K"
	defaultImagenSampleCount = 1
	maxGeminiInputImages     = 14
)

var (
	geminiAspectRatios = []string{"1:1", "1:4", "4:1", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9", "21:9", "8:1"}
	geminiImageSizes   = map[string]struct{}{"512": {}, "1K": {}, "2K": {}, "4K": {}}
	imagenAspectRatios = []string{"1:1", "3:4", "4:3", "9:16", "16:9"}
	imagenImageSizes   = map[string]struct{}{"1K": {}, "2K": {}}
)

// BuildGeminiPayload constructs a Google Gemini models/*:generateContent JSON body.
func BuildGeminiPayload(opts Options) ([]byte, error) {
	if strings.TrimSpace(opts.Prompt) == "" {
		return nil, ErrEmptyPrompt
	}

	parts := []map[string]any{{"text": opts.Prompt}}
	imageParts, err := geminiInputImageParts(opts)
	if err != nil {
		return nil, err
	}
	parts = append(parts, imageParts...)

	aspectRatio, imageSize := effectiveGeminiImageConfig(opts)
	payload := map[string]any{
		"contents": []map[string]any{
			{"role": "user", "parts": parts},
		},
		"generationConfig": map[string]any{
			"responseModalities": []string{"TEXT", "IMAGE"},
			"responseFormat": map[string]any{
				"image": map[string]any{
					"aspectRatio": aspectRatio,
					"imageSize":   imageSize,
					"mimeType":    outputMimeType(opts.OutputFormat),
				},
			},
		},
	}
	return encodeJSONPayload(payload)
}

// BuildImagenPayload constructs a Google Imagen models/*:predict JSON body.
func BuildImagenPayload(opts Options) ([]byte, error) {
	if strings.TrimSpace(opts.Prompt) == "" {
		return nil, ErrEmptyPrompt
	}

	aspectRatio, imageSize := effectiveImagenImageConfig(opts)
	parameters := map[string]any{
		"sampleCount": defaultImagenSampleCount,
		"imageSize":   imageSize,
		"aspectRatio": aspectRatio,
		"outputOptions": map[string]any{
			"mimeType": outputMimeType(opts.OutputFormat),
		},
	}
	if opts.ImagenSampleCount > 0 {
		parameters["sampleCount"] = opts.ImagenSampleCount
	}
	if strings.TrimSpace(opts.NegativePrompt) != "" {
		parameters["negativePrompt"] = opts.NegativePrompt
	}
	if opts.Seed != 0 {
		parameters["seed"] = opts.Seed
	}

	payload := map[string]any{
		"instances": []map[string]any{{"prompt": opts.Prompt}},
		"parameters": parameters,
	}
	return encodeJSONPayload(payload)
}

// EffectiveGeminiImageModel returns the configured Gemini image model or the provider default.
func EffectiveGeminiImageModel(opts Options) string {
	if strings.TrimSpace(opts.ImageModelID) != "" {
		return strings.TrimSpace(opts.ImageModelID)
	}
	return DefaultGeminiImageModel
}

// EffectiveImagenModel returns the configured Imagen model or the provider default.
func EffectiveImagenModel(opts Options) string {
	if strings.TrimSpace(opts.ImageModelID) != "" {
		return strings.TrimSpace(opts.ImageModelID)
	}
	return DefaultImagenModel
}

func effectiveGeminiImageConfig(opts Options) (aspectRatio, imageSize string) {
	aspectRatio, imageSize = deriveProviderImageConfig(opts.Size, geminiAspectRatios, geminiImageSizes, defaultGeminiAspectRatio, defaultGeminiImageSize)
	if isAllowedAspectRatio(opts.GeminiAspectRatio, geminiAspectRatios) {
		aspectRatio = strings.TrimSpace(opts.GeminiAspectRatio)
	}
	if isAllowedImageSize(opts.GeminiImageSize, geminiImageSizes) {
		imageSize = strings.ToUpper(strings.TrimSpace(opts.GeminiImageSize))
	}
	return aspectRatio, imageSize
}

func effectiveImagenImageConfig(opts Options) (aspectRatio, imageSize string) {
	aspectRatio, imageSize = deriveProviderImageConfig(opts.Size, imagenAspectRatios, imagenImageSizes, defaultImagenAspectRatio, defaultImagenImageSize)
	if isAllowedAspectRatio(opts.ImagenAspectRatio, imagenAspectRatios) {
		aspectRatio = strings.TrimSpace(opts.ImagenAspectRatio)
	}
	if isAllowedImageSize(opts.ImagenImageSize, imagenImageSizes) {
		imageSize = strings.ToUpper(strings.TrimSpace(opts.ImagenImageSize))
	}
	return aspectRatio, imageSize
}

func deriveProviderImageConfig(size string, allowedRatios []string, allowedSizes map[string]struct{}, fallbackRatio, fallbackSize string) (string, string) {
	width, height, ok := parseSizeValue(size)
	if !ok {
		return fallbackRatio, fallbackSize
	}
	return nearestAspectRatio(width, height, allowedRatios, fallbackRatio), imageSizeTier(width, height, allowedSizes, fallbackSize)
}

func nearestAspectRatio(width, height int, allowed []string, fallback string) string {
	if width <= 0 || height <= 0 {
		return fallback
	}
	target := float64(width) / float64(height)
	best := fallback
	bestDistance := math.Inf(1)
	for _, ratio := range allowed {
		value, ok := parseRatioValue(ratio)
		if !ok {
			continue
		}
		distance := math.Abs(math.Log(target / value))
		if distance < bestDistance {
			best = ratio
			bestDistance = distance
		}
	}
	return best
}

func imageSizeTier(width, height int, allowed map[string]struct{}, fallback string) string {
	longSide := width
	if height > longSide {
		longSide = height
	}
	candidate := fallback
	switch {
	case longSide >= 3200:
		candidate = "4K"
	case longSide >= 1800:
		candidate = "2K"
	case longSide <= 768:
		candidate = "512"
	default:
		candidate = "1K"
	}
	if _, ok := allowed[candidate]; ok {
		return candidate
	}
	if candidate == "4K" {
		if _, ok := allowed["2K"]; ok {
			return "2K"
		}
	}
	if candidate == "512" {
		if _, ok := allowed["1K"]; ok {
			return "1K"
		}
	}
	if _, ok := allowed[fallback]; ok {
		return fallback
	}
	return defaultGeminiImageSize
}

func parseRatioValue(ratio string) (float64, bool) {
	left, right, ok := strings.Cut(strings.TrimSpace(ratio), ":")
	if !ok {
		return 0, false
	}
	w, err := strconv.Atoi(left)
	if err != nil || w <= 0 {
		return 0, false
	}
	h, err := strconv.Atoi(right)
	if err != nil || h <= 0 {
		return 0, false
	}
	return float64(w) / float64(h), true
}

func isAllowedAspectRatio(value string, allowed []string) bool {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return false
	}
	for _, item := range allowed {
		if trimmed == item {
			return true
		}
	}
	return false
}

func isAllowedImageSize(value string, allowed map[string]struct{}) bool {
	trimmed := strings.ToUpper(strings.TrimSpace(value))
	if trimmed == "" {
		return false
	}
	_, ok := allowed[trimmed]
	return ok
}

func outputMimeType(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "jpeg", "jpg":
		return "image/jpeg"
	case "webp":
		return "image/webp"
	default:
		return "image/png"
	}
}

func geminiInputImageParts(opts Options) ([]map[string]any, error) {
	parts := make([]map[string]any, 0, len(opts.ImagePaths)+len(opts.EffectiveImageDataURLs()))
	for _, path := range opts.ImagePaths {
		trimmed := strings.TrimSpace(path)
		if trimmed == "" {
			continue
		}
		part, err := geminiInlineDataPartFromPath(trimmed)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
		if len(parts) > maxGeminiInputImages {
			return nil, fmt.Errorf("Gemini 图片生成最多支持 %d 张输入图片", maxGeminiInputImages)
		}
	}
	for _, dataURL := range opts.EffectiveImageDataURLs() {
		part, err := geminiInlineDataPartFromDataURL(dataURL)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
		if len(parts) > maxGeminiInputImages {
			return nil, fmt.Errorf("Gemini 图片生成最多支持 %d 张输入图片", maxGeminiInputImages)
		}
	}
	return parts, nil
}

func geminiInlineDataPartFromPath(path string) (map[string]any, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("找不到图片文件:%s", path)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("路径不是文件:%s", path)
	}
	if info.Size() > MaxInputImageBytes {
		return nil, fmt.Errorf("图片文件超过 50MB,请换一张更小的图片")
	}
	mimeType, ok := SupportedImageMime[strings.ToLower(filepath.Ext(path))]
	if !ok {
		return nil, fmt.Errorf("不支持的图片格式:%s", filepath.Ext(path))
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read image: %w", err)
	}
	return geminiInlineDataPart(mimeType, base64.StdEncoding.EncodeToString(raw)), nil
}

func geminiInlineDataPartFromDataURL(dataURL string) (map[string]any, error) {
	mimeType, payload, err := parseImageDataURL(dataURL)
	if err != nil {
		return nil, err
	}
	return geminiInlineDataPart(mimeType, payload), nil
}

func geminiInlineDataPart(mimeType, data string) map[string]any {
	return map[string]any{
		"inline_data": map[string]any{
			"mime_type": mimeType,
			"data":      data,
		},
	}
}

func parseImageDataURL(dataURL string) (mimeType, payload string, err error) {
	trimmed := strings.TrimSpace(dataURL)
	idx := strings.Index(trimmed, ",")
	if !strings.HasPrefix(trimmed, "data:") || idx < 0 {
		return "", "", fmt.Errorf("not a data URL")
	}
	header := trimmed[5:idx]
	payload = trimmed[idx+1:]
	if !strings.Contains(strings.ToLower(header), "base64") {
		return "", "", fmt.Errorf("data URL not base64")
	}
	mimeType = strings.TrimSpace(strings.Split(header, ";")[0])
	if _, ok := supportedMimeType(mimeType); !ok {
		return "", "", fmt.Errorf("不支持的图片 MIME:%s", mimeType)
	}
	raw, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return "", "", fmt.Errorf("decode data URL: %w", err)
	}
	if len(raw) > MaxInputImageBytes {
		return "", "", fmt.Errorf("图片文件超过 50MB,请换一张更小的图片")
	}
	return mimeType, payload, nil
}

func supportedMimeType(mimeType string) (string, bool) {
	for _, value := range SupportedImageMime {
		if mimeType == value {
			return value, true
		}
	}
	return "", false
}

func encodeJSONPayload(payload map[string]any) ([]byte, error) {
	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(payload); err != nil {
		return nil, fmt.Errorf("encode payload: %w", err)
	}
	return []byte(strings.TrimRight(buf.String(), "\n")), nil
}
