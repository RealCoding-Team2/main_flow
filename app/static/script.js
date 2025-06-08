document.addEventListener('DOMContentLoaded', function() {
    const chatArea  = document.getElementById('chat-area');
    const userInput = document.getElementById('user-input');
    const sendBtn   = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileUploadOptions = document.getElementById('file-upload-options');
    const closeUploadBtn    = document.getElementById('close-upload');
    const typingIndicator   = document.getElementById('typing-indicator');
    let conversationHistory = []; // 대화 기록 (필요시 LLM에 전달)

    // 채팅 스크롤 항상 하단 유지
    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // HTML 태그 이스케이프 함수 (XSS 방지)
    function escapeHtml(text) {
        if (typeof text !== 'string') {
            console.warn("escapeHtml: 입력값이 문자열이 아닙니다. 문자열로 변환합니다.", text);
            text = String(text);
        }
        return text.replace(/[&<>"']/g, function(m) {
            return ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            })[m];
        });
    }

    // 사용자 메시지를 채팅창에 추가하는 함수
    function addUserMessage(message) {
        const messageElement = `
            <div class="flex items-start space-x-3 justify-end my-2">
                <div class="max-w-xs md:max-w-md lg:max-w-lg bg-gray-200 rounded-xl p-3 shadow">
                    <p class="font-medium text-gray-800">사용자</p>
                    <p class="text-gray-700 mt-1 whitespace-pre-wrap">${escapeHtml(message)}</p>
                </div>
                <div class="bg-gray-300 text-gray-800 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-user"></i>
                </div>
            </div>
        `;
        chatArea.insertAdjacentHTML('beforeend', messageElement);
        scrollToBottom();
    }

    // AI 메시지를 채팅창에 추가하는 함수
    function addAiMessage(message, isAlert = false) {
        // 메시지 내의 줄바꿈(\n)을 <br> 태그로 변경하고, **텍스트**를 <strong>으로 변경
        let formattedMessage = escapeHtml(message)
                                .replace(/\n/g, '<br>')
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        const alertBgClass    = isAlert ? 'bg-red-50 border-l-4 border-red-500' : 'bg-blue-50';
        const titleColorClass = isAlert ? 'text-red-800 font-bold' : 'text-blue-800 font-medium';
        const iconBgClass     = isAlert ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800';


        const messageElement = `
            <div class="flex items-start space-x-3 my-2">
                <div class="${iconBgClass} rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="max-w-xs md:max-w-md lg:max-w-lg ${alertBgClass} rounded-xl p-3 shadow">
                    <p class="${titleColorClass}">보이스피싱 방지 AI</p>
                    <p class="text-gray-700 mt-1 whitespace-pre-wrap">${formattedMessage}</p>
                </div>
            </div>
        `;
        chatArea.insertAdjacentHTML('beforeend', messageElement);
        scrollToBottom();
    }

    // AI 메시지 중 이미지를 채팅창에 추가하는 함수
    function addAiMessage_img(imageSrc, boundingBox = null) {
        const canvasId = `canvas-${Date.now()}`;  // 고유 ID 부여

        const canvasContainer = document.createElement('div');
        canvasContainer.className = 'flex items-start space-x-3 my-2';
        canvasContainer.innerHTML = `
            <div class="bg-blue-100 text-blue-800 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                <i class="fas fa-robot"></i>
            </div>
            <div class="max-w-xs md:max-w-md lg:max-w-lg bg-blue-50 rounded-xl p-3 shadow">
                <p class="font-medium text-blue-800">보이스피싱 방지 AI</p>
                <canvas id="${canvasId}" class="mt-2 rounded-lg shadow-md max-w-full" style="width:100%; height:auto;"></canvas>
            </div>
        `;
        chatArea.appendChild(canvasContainer);

        const canvas = canvasContainer.querySelector('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // 만약, 서버로부터 박스좌표값을 받았다면 네모를 그림
            if (boundingBox) {
                const { x, y, w, h } = boundingBox;
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 4;
                ctx.strokeRect(x, y, w, h);
            }
            scrollToBottom();
        };
        img.src = imageSrc;
    }

    // 일반 텍스트 메시지 전송 처리 함수
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        addUserMessage(message);
        userInput.value = ''; // 입력창 비우기
        typingIndicator.classList.remove('hidden'); // 로딩 인디케이터 표시
        scrollToBottom();

        try {
            const response = await fetch('/api/chat', { // Flask의 /chat 라우트 호출
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) { // 서버 응답이 정상이 아닐 경우
                const errData = await response.json().catch(() => ({ error: `서버 응답 오류 (${response.status})` }));
                throw new Error(errData.error || `서버 오류 (${response.status})`);
            }

            const data = await response.json(); // 정상 응답 JSON 파싱
            addAiMessage(data.message || "AI로부터 응답을 받지 못했습니다."); // AI 응답 메시지 표시

        } catch (error) {
            console.error('Chat Error:', error);
            addAiMessage(`죄송합니다. 메시지 처리 중 오류가 발생했습니다: ${error.message}`, true);
        } finally {
            typingIndicator.classList.add('hidden'); // 로딩 인디케이터 숨김
            scrollToBottom();
        }
    }

    // 파일 업로드 처리 함수 (이미지/동영상)
    document.getElementById('media-upload').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            fileUploadOptions.classList.add('hidden'); // 파일 선택 옵션 숨김
            addUserMessage(`동영상/이미지 파일(${escapeHtml(file.name)})을 업로드했습니다.`);
            typingIndicator.classList.remove('hidden'); // 로딩 인디케이터 표시
            scrollToBottom();

            const formData = new FormData();
            formData.append('image', file); // 'image'라는 키로 파일 추가

            fetch('/api/deepfake/analyze_image', { // Flask의 딥페이크 분석 API 경로
                method: 'POST',
                body: formData,
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || `서버 분석 중 오류 발생 (${response.status})`);
                    }).catch(() => {
                        throw new Error(`서버 응답 오류 (${response.status})`);
                    });
                }
                return response.json();
            })
            .then(data => {
                typingIndicator.classList.add('hidden');
                console.log('전체 서버 응답 (data):', JSON.stringify(data, null, 2)); // 디버깅용 로그

                // 서버에서 응답이 왔다면, 이미지를 다시 띄우고, 좌표값이 있다면 그 좌표값대로 네모를 그립니다.
                addAiMessage_img(URL.createObjectURL(file), data.feature_analysis?.bounding_box);

                // --- 중요: 이제 data.llm_judgment는 이미 JSON 객체이므로 바로 사용합니다. ---
                if (data.llm_judgment) {
                    const probability = data.llm_judgment.deepfake_probability || '알 수 없음';
                    const reasoning = data.llm_judgment.reasoning || '제공되지 않음';
                    const recommendations = data.llm_judgment.recommendations_for_user || '추가 조언 없음';
                    const confidence = data.llm_judgment.confidence_score;

                    let resultMessage = `**분석 결과: ${probability}**\n`;
                    if (confidence !== undefined) {
                        resultMessage += `(판단 신뢰도: ${Math.round(confidence * 100)}%)\n\n`;
                    } else {
                        resultMessage += `\n`;
                    }
                    resultMessage += `**판단 근거:**\n${reasoning}\n\n`;
                    resultMessage += `**권장 사항:**\n${recommendations}`;

                    const isAlert = probability.includes("높음") || probability.includes("매우 높음");
                    addAiMessage(resultMessage.trim(), isAlert);

                } else if (data.error) {
                    addAiMessage(`분석 오류가 발생했습니다: ${data.error}`, true);
                } else {
                    addAiMessage("분석 결과를 받았으나, 예상치 못한 형식입니다. 관리자에게 문의해주세요.");
                }
            })
            .catch(error => { // fetch 자체의 실패 또는 이전 .then()에서 throw된 오류 처리
                typingIndicator.classList.add('hidden');
                console.error('Upload 또는 분석 처리 Error:', error);
                addAiMessage(`파일 업로드 또는 분석 처리 중 오류가 발생했습니다: ${error.message}`, true);
            })
            .finally(() => { // 성공/실패 여부와 관계없이 항상 실행
                 typingIndicator.classList.add('hidden');
                 scrollToBottom();
                 e.target.value = ''; // 파일 선택 초기화 (같은 파일 다시 업로드 가능하도록)
            });
        }
    });

    // 음성 파일 업로드 (데모 - 실제 기능 구현 필요)
    document.getElementById('audio-upload').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            fileUploadOptions.classList.add('hidden');
            addUserMessage(`음성 파일(${escapeHtml(file.name)})을 업로드했습니다.`);
            typingIndicator.classList.remove('hidden');
            scrollToBottom();
            
            // 실제 음성 파일 분석 API 호출
            const formData = new FormData();
            formData.append('audio', file); // 'audio'라는 키로 파일 추가

            fetch('/api/deepfake/analyze_audio', { // 음성 분석 API 경로
                method: 'POST',
                body: formData,
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || `서버 분석 중 오류 발생 (${response.status})`);
                    }).catch(() => {
                        throw new Error(`서버 응답 오류 (${response.status})`);
                    });
                }
                return response.json();
            })
            .then(data => {
                typingIndicator.classList.add('hidden');
                console.log('음성 분석 서버 응답:', JSON.stringify(data, null, 2));

                // STT 결과 표시
                if (data.stt_result && data.stt_result.success) {
                    const transcription = data.stt_result.transcription;
                    const confidence = data.stt_result.confidence;
                    const summaryInfo = data.stt_result.summary_info;
                    
                    // 요약된 텍스트가 있으면 그것을 사용, 없으면 원본 사용
                    const displayText = summaryInfo && summaryInfo.summarized_text ? 
                                      summaryInfo.summarized_text : transcription;
                    
                    let sttMessage = `**음성 변환 및 요약 결과:**\n"${displayText}"\n`;
                    if (confidence !== undefined) {
                        sttMessage += `(변환 신뢰도: ${Math.round(confidence * 100)}%)\n`;
                    }
                    
                    // 요약 정보 추가
                    if (summaryInfo) {
                        if (summaryInfo.is_summarized) {
                            sttMessage += `\n**텍스트 처리:** 원본 ${summaryInfo.original_length}자에서 ${summaryInfo.summarized_length}자로 요약됨\n`;
                        } else {
                            sttMessage += `\n**텍스트 처리:** ${summaryInfo.summary_reason}\n`;
                        }
                    }
                    
                    addAiMessage(sttMessage);
                }

                // 보이스피싱 분석 결과 표시
                if (data.llm_judgment) {
                    const probability = data.llm_judgment.voicephishing_probability || '알 수 없음';
                    const reasoning = data.llm_judgment.reasoning || '제공되지 않음';
                    const recommendations = data.llm_judgment.recommendations_for_user || '추가 조언 없음';
                    const confidence = data.llm_judgment.confidence_score;
                    const riskLevel = data.llm_judgment.risk_level;
                    const detectedPatterns = data.llm_judgment.detected_patterns;
                    const textProcessing = data.llm_judgment.text_processing;

                    let resultMessage = `**보이스피싱 분석 결과: ${probability}**\n`;
                    if (riskLevel) {
                        resultMessage += `**위험도:** ${riskLevel}\n`;
                    }
                    if (confidence !== undefined) {
                        resultMessage += `**분석 신뢰도:** ${Math.round(confidence * 100)}%\n`;
                    }
                    
                    // 텍스트 처리 정보 추가
                    if (textProcessing) {
                        if (textProcessing.was_summarized) {
                            resultMessage += `**분석 방식:** 요약된 텍스트 기반 분석 (${textProcessing.original_text_length}자 → ${textProcessing.analyzed_text_length}자)\n`;
                        } else {
                            resultMessage += `**분석 방식:** 원본 텍스트 직접 분석\n`;
                        }
                    }
                    
                    resultMessage += `\n**분석 근거:**\n${reasoning}\n\n`;
                    
                    if (detectedPatterns && detectedPatterns.length > 0) {
                        resultMessage += `**감지된 의심 패턴:**\n`;
                        detectedPatterns.forEach(pattern => {
                            resultMessage += `• ${pattern}\n`;
                        });
                        resultMessage += `\n`;
                    }
                    
                    resultMessage += `**권장 조치:**\n${recommendations}`;

                    // 위험도에 따라 알림 스타일 결정
                    const isAlert = probability.includes("높음") || riskLevel === "고위험";
                    addAiMessage(resultMessage.trim(), isAlert);

                } else if (data.error) {
                    addAiMessage(`분석 오류가 발생했습니다: ${data.error}`, true);
                } else {
                    addAiMessage("분석 결과를 받았으나, 예상치 못한 형식입니다. 관리자에게 문의해주세요.");
                }
            })
            .catch(error => {
                typingIndicator.classList.add('hidden');
                console.error('음성 업로드 또는 분석 처리 Error:', error);
                addAiMessage(`음성 파일 업로드 또는 분석 처리 중 오류가 발생했습니다: ${error.message}`, true);
            })
            .finally(() => {
                typingIndicator.classList.add('hidden');
                scrollToBottom();
                e.target.value = ''; // 파일 선택 초기화 (같은 파일 다시 업로드 가능하도록)
            });
        }
    });

    // 이벤트 리스너 등록
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault(); // Enter 키 기본 동작(줄바꿈 등) 방지
            sendMessage();
        }
    });

    uploadBtn.addEventListener('click', function() {
        fileUploadOptions.classList.toggle('hidden');
    });

    closeUploadBtn.addEventListener('click', function() {
        fileUploadOptions.classList.add('hidden');
    });

});