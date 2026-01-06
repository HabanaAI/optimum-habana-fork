# /v1/images/edits 接口参数详细说明

本接口用于根据文本描述对图片进行编辑。以下为实际用到的参数说明：

## 1. image (必填)
- **说明**：待编辑的原始图片，支持单张或多张图片上传。
- **类型**：UploadFile 或 List[UploadFile]
- **传递方式**：multipart/form-data，字段名为 image 或 image[]
- **示例**：通过表单上传图片文件

## 2. prompt (str, 必填)
- **说明**：编辑图片的文本描述（如“给图片加上蓝天白云”）。
- **示例**：
  ```json
  "prompt": "add blue sky and white clouds"
  ```

## 3. quality (str, 可选)
- **说明**：图片生成质量，影响推理步数。
  - 可选值："high"（步数40）、"medium"（步数20）、"low"（步数10）。
  - 未指定时默认步数为20。
- **示例**：
  ```json
  "quality": "high"
  ```

### quality 与实际步数（num_inference_steps）对应关系：

| quality  | num_inference_steps |
|----------|--------------------|
| high     | 40                 |
| medium   | 20                 |
| low      | 10                 |
| 未指定或其它 | 20                 |

## 4. size (str, 可选)
- **说明**：输出图片尺寸，格式为 "宽x高"，如 "1024x1024"。
- **示例**：
  ```json
  "size": "1024x1536"
  ```

## 5. n (int, 可选)
- **说明**：每张输入图片生成的图片数量，默认为 1。
- **示例**：
  ```json
  "n": 2
  ```


## curl 示例

以下是一个完整的 curl 命令示例，演示如何调用 /v1/images/edits 接口：

```bash
curl -X POST \
  http://localhost:9390/v1/images/edits \
  -F "image=@/path/to/your/image.jpg" \
  -F "prompt=add blue sky and white clouds" \
  -F "quality=high" \
  -F "size=1024x1536" \
  -F "n=2"
```

- 请将 `/path/to/your/image.jpg` 替换为实际图片路径。
- 可根据需要省略可选参数（如 quality、size、n）。

---

- 这些参数均通过 multipart/form-data 方式传递。
- 未用到的参数无需传递。
