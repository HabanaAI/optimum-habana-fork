# /v1/images/generations 接口参数详细说明

本接口用于根据文本描述生成图片。以下为实际用到的参数说明：

## 1. prompt (str, 必填)
- **说明**：用于描述希望生成的图片内容的文本提示（prompt）。
- **示例**：
  ```json
  "prompt": "a cat sitting on a bench in the park"
  ```

## 2. quality (str, 可选)
  - 可选值："high"（高质量，步数多）、"medium"（中等质量）、"low"（低质量，步数少）。
  - 未指定时使用默认步数。
  ```json
  "quality": "high"
  ```

### quality 与实际步数（num_inference_steps）对应关系：

#### Tongyi-MAI/Z-Image-Turbo：

| quality  | num_inference_steps |
|----------|--------------------|
| high     | 20                 |
| medium   | 9                  |
| low      | 5                  |
| 未指定或其它 | 9                  |

#### 其他模型:

| quality  | num_inference_steps |
|----------|--------------------|
| high     | 50                 |
| medium   | 25                 |
| low      | 10                 |
| 未指定或其它 | 25                 |

## 3. size (str, 可选)
- **说明**：图片尺寸，格式为 "宽x高"，如 "1024x1024"。会被解析为 width 和 height。
- **示例**：
  ```json
  "size": "1024x1536"
  ```

## 4. n (int, 可选)
- **说明**：生成图片的数量。默认为 1。
- **示例**：
  ```json
  "n": 2
  ```

---

- 这些参数均为 JSON body 传递。
- 未用到的参数无需传递。

## curl 示例

以下是一个完整的 curl 命令示例，演示如何调用 /v1/images/generations 接口：

```bash
curl -X POST \
  http://localhost:9391/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat sitting on a bench in the park",
    "quality": "high",
    "size": "1024x1536",
    "n": 2
  }'
```

- 可根据需要省略可选参数（如 quality、size、n）。
