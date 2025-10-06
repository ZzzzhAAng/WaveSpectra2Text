#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音识别API服务
提供REST API接口，接收音频文件并返回识别结果
"""

from flask import Flask, request, jsonify, render_template_string
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
import time
import logging

from production_inference_demo import ProductionSpeechRecognizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局识别器实例
recognizer = None

# 支持的音频格式
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

def allowed_file(filename):
    """检查文件格式是否支持"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML模板 - 简单的上传界面
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>语音识别服务</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: 0 auto; }
        .upload-area { 
            border: 2px dashed #ccc; 
            border-radius: 10px; 
            padding: 40px; 
            text-align: center; 
            margin: 20px 0;
        }
        .result { 
            background: #f0f8ff; 
            padding: 20px; 
            border-radius: 5px; 
            margin: 20px 0;
        }
        button { 
            background: #4CAF50; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer;
        }
        button:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 语音识别服务</h1>
        <p>上传音频文件，获取识别结果</p>
        
        <div class="upload-area">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="audioFile" name="audio" accept=".wav,.mp3,.flac,.m4a,.ogg" required>
                <br><br>
                <button type="submit">🎯 开始识别</button>
            </form>
        </div>
        
        <div id="result" style="display:none;"></div>
        
        <h3>📋 API使用说明</h3>
        <p><strong>POST /api/recognize</strong></p>
        <pre>
curl -X POST -F "audio=@your_audio.wav" http://localhost:5000/api/recognize
        </pre>
        
        <h3>📊 支持格式</h3>
        <p>WAV, MP3, FLAC, M4A, OGG</p>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('audioFile');
            const resultDiv = document.getElementById('result');
            
            if (!fileInput.files[0]) {
                alert('请选择音频文件');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio', fileInput.files[0]);
            
            resultDiv.innerHTML = '<p>🔄 正在识别中...</p>';
            resultDiv.style.display = 'block';
            
            try {
                const response = await fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultDiv.innerHTML = `
                        <h3>🎉 识别结果</h3>
                        <p><strong>识别文本:</strong> "${result.text}"</p>
                        <p><strong>处理时间:</strong> ${result.processing_time.total.toFixed(3)}秒</p>
                        <p><strong>详细信息:</strong></p>
                        <ul>
                            <li>预处理: ${result.processing_time.preprocessing.toFixed(3)}秒</li>
                            <li>推理: ${result.processing_time.inference.toFixed(3)}秒</li>
                            <li>频谱形状: ${result.spectrogram_shape}</li>
                        </ul>
                    `;
                } else {
                    resultDiv.innerHTML = `<p>❌ 识别失败: ${result.error}</p>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<p>❌ 请求失败: ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """主页 - 上传界面"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/recognize', methods=['POST'])
def recognize_audio():
    """API接口 - 音频识别"""
    try:
        # 检查是否有文件上传
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有上传音频文件'
            }), 400
        
        file = request.files['audio']
        
        # 检查文件名
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            }), 400
        
        # 检查文件格式
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'不支持的文件格式，支持格式: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # 保存临时文件
        filename = secure_filename(file.filename)
        temp_id = str(uuid.uuid4())
        temp_filename = f"{temp_id}_{filename}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        logger.info(f"处理音频文件: {filename} (临时文件: {temp_path})")
        
        # 使用识别器处理音频
        result = recognizer.process_new_audio(temp_path, show_details=False)
        
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # 添加文件信息
        result['filename'] = filename
        result['file_size'] = len(file.read()) if hasattr(file, 'read') else 'unknown'
        file.seek(0)  # 重置文件指针
        
        logger.info(f"识别完成: {filename} -> '{result.get('text', 'N/A')}'")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API处理错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': recognizer is not None,
        'timestamp': time.time()
    })

@app.route('/api/info', methods=['GET'])
def get_info():
    """获取服务信息"""
    return jsonify({
        'service': '语音识别API',
        'version': '1.0.0',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB',
        'model_info': {
            'vocab_size': 14,
            'supported_languages': ['中文数字'],
            'model_type': 'Transformer Encoder-Decoder'
        }
    })

def initialize_recognizer(model_path, device='cpu'):
    """初始化识别器"""
    global recognizer
    
    try:
        logger.info(f"初始化语音识别器: {model_path}")
        recognizer = ProductionSpeechRecognizer(model_path, device)
        logger.info("识别器初始化成功")
        return True
    except Exception as e:
        logger.error(f"识别器初始化失败: {e}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='语音识别API服务')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--host', type=str, default='localhost', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    print("🚀 启动语音识别API服务")
    print("=" * 50)
    
    # 初始化识别器
    if not initialize_recognizer(args.model, args.device):
        print("❌ 识别器初始化失败，服务无法启动")
        return
    
    print(f"✅ 服务配置:")
    print(f"  模型路径: {args.model}")
    print(f"  服务地址: http://{args.host}:{args.port}")
    print(f"  计算设备: {args.device}")
    print(f"  调试模式: {args.debug}")
    
    print(f"\n📋 API端点:")
    print(f"  主页: http://{args.host}:{args.port}/")
    print(f"  识别API: http://{args.host}:{args.port}/api/recognize")
    print(f"  健康检查: http://{args.host}:{args.port}/api/health")
    print(f"  服务信息: http://{args.host}:{args.port}/api/info")
    
    print(f"\n💡 使用方法:")
    print(f"  1. 浏览器访问: http://{args.host}:{args.port}")
    print(f"  2. 上传音频文件获取识别结果")
    print(f"  3. 或使用API: curl -X POST -F \"audio=@file.wav\" http://{args.host}:{args.port}/api/recognize")
    
    # 启动服务
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")

if __name__ == "__main__":
    main()