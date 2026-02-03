#!/bin/bash

# SongGeneration API 自动测试脚本
# API地址: http://127.0.0.1:8485
# 测试结果保存目录: test_results

# 设置代理

set -e

API_BASE="http://127.0.0.1:8486"
RESULTS_DIR="test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/test_results_$TIMESTAMP.log"

# 创建测试结果目录
mkdir -p "$RESULTS_DIR"

echo "========================================" | tee -a "$RESULTS_FILE"
echo "SongGeneration API 自动测试" | tee -a "$RESULTS_FILE"
echo "API地址: $API_BASE" | tee -a "$RESULTS_FILE"
echo "测试时间: $(date)" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# 测试计数器
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 测试函数
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expected_status="${3:-200}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo "----------------------------------------" | tee -a "$RESULTS_FILE"
    echo "测试 #$TOTAL_TESTS: $test_name" | tee -a "$RESULTS_FILE"
    echo "命令: $test_cmd" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    
    # 执行测试命令
    local response
    local http_code
    
    # 执行命令并捕获输出和HTTP状态码
    response=$(eval "$test_cmd" 2>&1)
    http_code=$(eval "${test_cmd} -w '%{http_code}' -o /dev/null 2>&1" || echo "000")
    
    echo "HTTP状态码: $http_code" | tee -a "$RESULTS_FILE"
    echo "响应内容:" | tee -a "$RESULTS_FILE"
    echo "$response" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    
    # 检查是否成功
    if [ "$http_code" == "$expected_status" ] || [ "$http_code" == "200" ]; then
        echo "✓ 测试通过" | tee -a "$RESULTS_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo "✗ 测试失败 (期望状态码: $expected_status, 实际: $http_code)" | tee -a "$RESULTS_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# 存储任务ID
TASK_ID=""
TASK_ID_SEPARATE=""

echo "开始API测试..." | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# ==================== 测试1: 查询自动提示音频类型 ====================
echo "【测试组1: 查询API】" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

run_test "查询自动提示音频类型" \
    "curl -s $API_BASE/v1/audio/song/query/auto_prompt_audio_type"

echo "" | tee -a "$RESULTS_FILE"

# ==================== 测试2: 基本生成请求 (仅gt_lyric) ====================
echo "【测试组2: 基本生成请求】" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

TEST1_RESPONSE=$(curl -s -X POST "$API_BASE/v1/audio/song" \
    --form-string gt_lyric="[intro-short] ; [verse] These faded memories of us. I can't erase the tears you cried before. Unchained this heart to find its way. My peace won't beg you to stay ; [chorus] Like a fool begs for supper. I find myself waiting for her. Only to find the broken pieces of my heart. That was needed for my soul to love again ; [outro-short]")

echo "测试: 基本生成请求 (仅gt_lyric)" | tee -a "$RESULTS_FILE"
echo "命令: curl -X POST $API_BASE/v1/audio/song --form-string gt_lyric=..." | tee -a "$RESULTS_FILE"
echo "响应: $TEST1_RESPONSE" | tee -a "$RESULTS_FILE"

# 提取任务ID
TASK_ID=$(echo "$TEST1_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
if [ -n "$TASK_ID" ]; then
    echo "✓ 测试通过 - 任务ID: $TASK_ID" | tee -a "$RESULTS_FILE"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "✗ 测试失败 - 无法获取任务ID" | tee -a "$RESULTS_FILE"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo "" | tee -a "$RESULTS_FILE"

# ==================== 测试3: 带descriptions的生成请求 ====================
echo "测试: 带descriptions的生成请求" | tee -a "$RESULTS_FILE"

TEST2_RESPONSE=$(curl -s -X POST "$API_BASE/v1/audio/song" \
    --form-string gt_lyric="[intro-short] ; [verse] These faded memories of us. I can't erase the tears you cried before. Unchained this heart to find its way. My peace won't beg you to stay ; [chorus] Like a fool begs for supper. I find myself waiting for her. Only to find the broken pieces of my heart. That was needed for my soul to love again ; [outro-short]" \
    --form-string descriptions="female, dark, pop, sad, piano and drums")

echo "命令: curl -X POST $API_BASE/v1/audio/song --form-string gt_lyric=... --form-string descriptions=..." | tee -a "$RESULTS_FILE"
echo "响应: $TEST2_RESPONSE" | tee -a "$RESULTS_FILE"

TASK_ID_2=$(echo "$TEST2_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
if [ -n "$TASK_ID_2" ]; then
    echo "✓ 测试通过 - 任务ID: $TASK_ID_2" | tee -a "$RESULTS_FILE"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "✗ 测试失败 - 无法获取任务ID" | tee -a "$RESULTS_FILE"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo "" | tee -a "$RESULTS_FILE"

# ==================== 测试4: 带auto_prompt_audio_type的生成请求 ====================
echo "测试: 带auto_prompt_audio_type的生成请求" | tee -a "$RESULTS_FILE"

TEST3_RESPONSE=$(curl -s -X POST "$API_BASE/v1/audio/song" \
    --form-string gt_lyric="[intro-short] ; [verse] These faded memories of us. I can't erase the tears you cried before. Unchained this heart to find its way. My peace won't beg you to stay ; [chorus] Like a fool begs for supper. I find myself waiting for her. Only to find the broken pieces of my heart. That was needed for my soul to love again ; [outro-short]" \
    --form-string auto_prompt_audio_type="Metal")

echo "命令: curl -X POST $API_BASE/v1/audio/song --form-string gt_lyric=... --form-string auto_prompt_audio_type=Metal" | tee -a "$RESULTS_FILE"
echo "响应: $TEST3_RESPONSE" | tee -a "$RESULTS_FILE"

TASK_ID_3=$(echo "$TEST3_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
if [ -n "$TASK_ID_3" ]; then
    echo "✓ 测试通过 - 任务ID: $TASK_ID_3" | tee -a "$RESULTS_FILE"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "✗ 测试失败 - 无法获取任务ID" | tee -a "$RESULTS_FILE"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo "" | tee -a "$RESULTS_FILE"

# ==================== 测试5: 带gen_type=separate的生成请求 ====================
echo "测试: 带gen_type=separate的生成请求" | tee -a "$RESULTS_FILE"

TEST4_RESPONSE=$(curl -s -X POST "$API_BASE/v1/audio/song" \
    --form-string gt_lyric="[intro-short] ; [verse] These faded memories of us. I can't erase the tears you cried before. Unchained this heart to find its way. My peace won't beg you to stay ; [chorus] Like a fool begs for supper. I find myself waiting for her. Only to find the broken pieces of my heart. That was needed for my soul to love again ; [outro-short]" \
    --form-string gen_type="separate")

echo "命令: curl -X POST $API_BASE/v1/audio/song --form-string gt_lyric=... --form-string gen_type=separate" | tee -a "$RESULTS_FILE"
echo "响应: $TEST4_RESPONSE" | tee -a "$RESULTS_FILE"

TASK_ID_SEPARATE=$(echo "$TEST4_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
if [ -n "$TASK_ID_SEPARATE" ]; then
    echo "✓ 测试通过 - 任务ID: $TASK_ID_SEPARATE (将生成独立音轨)" | tee -a "$RESULTS_FILE"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "✗ 测试失败 - 无法获取任务ID" | tee -a "$RESULTS_FILE"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo "" | tee -a "$RESULTS_FILE"

# ==================== 测试6: 中文歌词生成请求 ====================
echo "测试: 中文歌词生成请求" | tee -a "$RESULTS_FILE"

TEST5_RESPONSE=$(curl -s -X POST "$API_BASE/v1/audio/song" \
    --form-string gt_lyric="[intro-short] ; [verse] 花朵绽放如诗篇.随风轻舞动.湖面波光映倒影.心随波浪荡漾中.感受着这美好时光.仿佛梦境般飘渺 ; [chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅.阳光洒满湖面.一切都如此美好.仿佛置身人间天堂.让人心醉神迷 ; [outro-short]")

echo "命令: curl -X POST $API_BASE/v1/audio/song --form-string gt_lyric=..." | tee -a "$RESULTS_FILE"
echo "响应: $TEST5_RESPONSE" | tee -a "$RESULTS_FILE"

TASK_ID_5=$(echo "$TEST5_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
if [ -n "$TASK_ID_5" ]; then
    echo "✓ 测试通过 - 任务ID: $TASK_ID_5" | tee -a "$RESULTS_FILE"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "✗ 测试失败 - 无法获取任务ID" | tee -a "$RESULTS_FILE"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo "" | tee -a "$RESULTS_FILE"

# ==================== 测试7: 带prompt_audio的生成请求 (如果存在示例文件) ====================
if [ -f "sample/sample_prompt_audio.wav" ]; then
    echo "测试: 带prompt_audio的生成请求" | tee -a "$RESULTS_FILE"
    
    TEST6_RESPONSE=$(curl -s -X POST "$API_BASE/v1/audio/song" \
        --form-string gt_lyric="[intro-short] ; [verse] These faded memories of us. I can't erase the tears you cried before. Unchained this heart to find its way. My peace won't beg you to stay ; [chorus] Like a fool begs for supper. I find myself waiting for her. Only to find the broken pieces of my heart. That was needed for my soul to love again ; [outro-short]" \
        -F prompt_audio="@sample/sample_prompt_audio.wav")
    
    echo "命令: curl -X POST $API_BASE/v1/audio/song --form-string gt_lyric=... -F prompt_audio=@sample/sample_prompt_audio.wav" | tee -a "$RESULTS_FILE"
    echo "响应: $TEST6_RESPONSE" | tee -a "$RESULTS_FILE"
    
    TASK_ID_6=$(echo "$TEST6_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    if [ -n "$TASK_ID_6" ]; then
        echo "✓ 测试通过 - 任务ID: $TASK_ID_6" | tee -a "$RESULTS_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "✗ 测试失败 - 无法获取任务ID" | tee -a "$RESULTS_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo "" | tee -a "$RESULTS_FILE"
else
    echo "⚠ 跳过测试: 带prompt_audio的生成请求 (sample/sample_prompt_audio.wav 不存在)" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
fi

# ==================== 测试组3: 查询任务状态 ====================
echo "【测试组3: 任务状态查询】" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

if [ -n "$TASK_ID" ]; then
    # 查询任务状态
    sleep 2
    
    echo "测试: 查询任务状态 (任务ID: $TASK_ID)" | tee -a "$RESULTS_FILE"
    STATUS_RESPONSE=$(curl -s "$API_BASE/v1/audio/song/$TASK_ID")
    echo "命令: curl -s $API_BASE/v1/audio/song/$TASK_ID" | tee -a "$RESULTS_FILE"
    echo "响应: $STATUS_RESPONSE" | tee -a "$RESULTS_FILE"
    
    if echo "$STATUS_RESPONSE" | grep -q "\"id\":\"$TASK_ID\""; then
        echo "✓ 测试通过" | tee -a "$RESULTS_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "✗ 测试失败" | tee -a "$RESULTS_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo "" | tee -a "$RESULTS_FILE"
    
    # 检查任务状态并等待完成
    echo "等待任务完成..." | tee -a "$RESULTS_FILE"
    MAX_WAIT=120  # 最大等待120秒
    WAIT_TIME=0
    
    while [ $WAIT_TIME -lt $MAX_WAIT ]; do
        STATUS_RESPONSE=$(curl -s "$API_BASE/v1/audio/song/$TASK_ID")
        TASK_STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        
        echo "  当前状态: $TASK_STATUS (等待 ${WAIT_TIME}s)" | tee -a "$RESULTS_FILE"
        
        if [ "$TASK_STATUS" == "completed" ]; then
            echo "✓ 任务已完成" | tee -a "$RESULTS_FILE"
            break
        elif [ "$TASK_STATUS" == "error" ]; then
            echo "✗ 任务执行出错" | tee -a "$RESULTS_FILE"
            break
        fi
        
        sleep 5
        WAIT_TIME=$((WAIT_TIME + 5))
    done
    
    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo "⚠ 等待超时，任务可能仍在进行中" | tee -a "$RESULTS_FILE"
    fi
    echo "" | tee -a "$RESULTS_FILE"
    
    # ==================== 测试组4: 下载音频 ====================
    if [ "$TASK_STATUS" == "completed" ]; then
        echo "【测试组4: 下载音频】" | tee -a "$RESULTS_FILE"
        echo "" | tee -a "$RESULTS_FILE"
        
        # 下载完整音频
        echo "测试: 下载完整音频" | tee -a "$RESULTS_FILE"
        curl -s "$API_BASE/v1/audio/song/$TASK_ID/content" -o "$RESULTS_DIR/test_audio_$TASK_ID.wav" -w "HTTP状态码: %{http_code}\n"
        if [ -f "$RESULTS_DIR/test_audio_$TASK_ID.wav" ] && [ -s "$RESULTS_DIR/test_audio_$TASK_ID.wav" ]; then
            echo "✓ 测试通过 - 音频已保存到: $RESULTS_DIR/test_audio_$TASK_ID.wav" | tee -a "$RESULTS_FILE"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo "✗ 测试失败 - 音频下载失败" | tee -a "$RESULTS_FILE"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        echo "" | tee -a "$RESULTS_FILE"
        
        # 下载BGM音频 (仅当gen_type=separate时)
        if [ -n "$TASK_ID_SEPARATE" ]; then
            sleep 2
            
            # 先检查分离任务的状态
            echo "等待分离任务完成..." | tee -a "$RESULTS_FILE"
            SEP_WAIT=0
            while [ $SEP_WAIT -lt $MAX_WAIT ]; do
                SEP_STATUS_RESPONSE=$(curl -s "$API_BASE/v1/audio/song/$TASK_ID_SEPARATE")
                SEP_TASK_STATUS=$(echo "$SEP_STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
                
                echo "  分离任务状态: $SEP_TASK_STATUS (等待 ${SEP_WAIT}s)" | tee -a "$RESULTS_FILE"
                
                if [ "$SEP_TASK_STATUS" == "completed" ]; then
                    echo "✓ 分离任务已完成" | tee -a "$RESULTS_FILE"
                    break
                elif [ "$SEP_TASK_STATUS" == "error" ]; then
                    echo "✗ 分离任务执行出错" | tee -a "$RESULTS_FILE"
                    break
                fi
                
                sleep 5
                SEP_WAIT=$((SEP_WAIT + 5))
            done
            
            if [ "$SEP_TASK_STATUS" == "completed" ]; then
                echo "测试: 下载BGM音频" | tee -a "$RESULTS_FILE"
                curl -s "$API_BASE/v1/audio/song/$TASK_ID_SEPARATE/bgm/content" -o "$RESULTS_DIR/test_bgm_$TASK_ID_SEPARATE.wav" -w "HTTP状态码: %{http_code}\n"
                if [ -f "$RESULTS_DIR/test_bgm_$TASK_ID_SEPARATE.wav" ] && [ -s "$RESULTS_DIR/test_bgm_$TASK_ID_SEPARATE.wav" ]; then
                    echo "✓ 测试通过 - BGM音频已保存到: $RESULTS_DIR/test_bgm_$TASK_ID_SEPARATE.wav" | tee -a "$RESULTS_FILE"
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                else
                    echo "✗ 测试失败 - BGM音频下载失败" | tee -a "$RESULTS_FILE"
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                fi
                TOTAL_TESTS=$((TOTAL_TESTS + 1))
                echo "" | tee -a "$RESULTS_FILE"
                
                # 下载Vocal音频
                echo "测试: 下载Vocal音频" | tee -a "$RESULTS_FILE"
                curl -s "$API_BASE/v1/audio/song/$TASK_ID_SEPARATE/vocal/content" -o "$RESULTS_DIR/test_vocal_$TASK_ID_SEPARATE.wav" -w "HTTP状态码: %{http_code}\n"
                if [ -f "$RESULTS_DIR/test_vocal_$TASK_ID_SEPARATE.wav" ] && [ -s "$RESULTS_DIR/test_vocal_$TASK_ID_SEPARATE.wav" ]; then
                    echo "✓ 测试通过 - Vocal音频已保存到: $RESULTS_DIR/test_vocal_$TASK_ID_SEPARATE.wav" | tee -a "$RESULTS_FILE"
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                else
                    echo "✗ 测试失败 - Vocal音频下载失败" | tee -a "$RESULTS_FILE"
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                fi
                TOTAL_TESTS=$((TOTAL_TESTS + 1))
                echo "" | tee -a "$RESULTS_FILE"
            else
                echo "⚠ 分离任务未完成，跳过BGM和Vocal下载测试" | tee -a "$RESULTS_FILE"
                echo "" | tee -a "$RESULTS_FILE"
            fi
        fi
    else
        echo "⚠ 任务未完成，跳过音频下载测试" | tee -a "$RESULTS_FILE"
        echo "" | tee -a "$RESULTS_FILE"
    fi
    
    # ==================== 测试组5: 删除任务 ====================
    echo "【测试组5: 删除任务】" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    
    # 注意: 只能删除queued状态的任务
    # 重新创建一个任务来测试删除功能
    echo "创建新任务用于删除测试..." | tee -a "$RESULTS_FILE"
    DELETE_TEST_RESPONSE=$(curl -s -X POST "$API_BASE/v1/audio/song" \
        --form-string gt_lyric="[intro-short] ; [verse] Test for deletion. This is a test lyric ; [outro-short]")
    
    DELETE_TASK_ID=$(echo "$DELETE_TEST_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    
    if [ -n "$DELETE_TASK_ID" ]; then
        echo "新任务ID: $DELETE_TASK_ID" | tee -a "$RESULTS_FILE"
        sleep 1
        
        # 立即尝试删除
        echo "测试: 删除任务 (任务ID: $DELETE_TASK_ID)" | tee -a "$RESULTS_FILE"
        DELETE_RESPONSE=$(curl -s "$API_BASE/v1/audio/song/$DELETE_TASK_ID/delete")
        echo "命令: curl -s $API_BASE/v1/audio/song/$DELETE_TASK_ID/delete" | tee -a "$RESULTS_FILE"
        echo "响应: $DELETE_RESPONSE" | tee -a "$RESULTS_FILE"
        
        if echo "$DELETE_RESPONSE" | grep -q '"success"'; then
            echo "✓ 测试通过 - 任务已删除" | tee -a "$RESULTS_FILE"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo "✗ 测试失败 - 删除可能失败 (任务可能已开始执行)" | tee -a "$RESULTS_FILE"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    else
        echo "✗ 无法创建测试任务，跳过删除测试" | tee -a "$RESULTS_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    fi
    echo "" | tee -a "$RESULTS_FILE"
else
    echo "⚠ 没有有效的任务ID，跳过后续测试" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
fi

# ==================== 测试总结 ====================
echo "========================================" | tee -a "$RESULTS_FILE"
echo "测试总结" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "总测试数: $TOTAL_TESTS" | tee -a "$RESULTS_FILE"
echo "通过: $PASSED_TESTS" | tee -a "$RESULTS_FILE"
echo "失败: $FAILED_TESTS" | tee -a "$RESULTS_FILE"
echo "通过率: $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "测试结果已保存到: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "测试完成时间: $(date)" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"

# 显示下载的音频文件
echo "" | tee -a "$RESULTS_FILE"
echo "生成的音频文件:" | tee -a "$RESULTS_FILE"
ls -lh "$RESULTS_DIR"/*.wav 2>/dev/null | tee -a "$RESULTS_FILE" || echo "无音频文件" | tee -a "$RESULTS_FILE"

# 返回退出码
if [ $FAILED_TESTS -eq 0 ]; then
    echo ""
    echo "✓ 所有测试通过!"
    exit 0
else
    echo ""
    echo "✗ 部分测试失败，请查看日志"
    exit 1
fi
