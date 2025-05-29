document.addEventListener('DOMContentLoaded', function() {
    const chatArea = document.getElementById('chat-area');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileUploadOptions = document.getElementById('file-upload-options');
    const closeUploadBtn = document.getElementById('close-upload');
    const typingIndicator = document.getElementById('typing-indicator');
    let conversationHistory = [];

    // 채팅 스크롤 항상 하단 유지
    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // 사용자 메시지 추가
    function addUserMessage(message) {
        const messageElement = `
            <div class="flex items-start space-x-3 justify-end">
                <div class="max-w-xs md:max-w-md lg:max-w-lg bg-gray-200 rounded-xl p-3">
                    <p class="font-medium text-gray-800">사용자</p>
                    <p class="text-gray-700 mt-1">${escapeHtml(message)}</p>
                </div>
                <div class="bg-gray-300 text-gray-800 rounded-full w-10 h-10 flex items-center justify-center">
                    <i class="fas fa-user"></i>
                </div>
            </div>
        `;
        chatArea.insertAdjacentHTML('beforeend', messageElement);
        scrollToBottom();
    }

    // AI 메시지 추가
    function addAiMessage(message) {
        const messageElement = `
            <div class="flex items-start space-x-3">
                <div class="bg-blue-100 text-blue-800 rounded-full w-10 h-10 flex items-center justify-center">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="max-w-xs md:max-w-md lg:max-w-lg bg-blue-50 rounded-xl p-3">
                    <p class="font-medium text-blue-800">보이스피싱 방지 AI</p>
                    <p class="text-gray-700 mt-1">${escapeHtml(message)}</p>
                </div>
            </div>
        `;
        chatArea.insertAdjacentHTML('beforeend', messageElement);
        scrollToBottom();
    }

    // AI 메시지 추가
    function addAiMessage_img(message) {
        const messageElement = `
            <div class="flex items-start space-x-3">
                <div class="bg-blue-100 text-blue-800 rounded-full w-10 h-10 flex items-center justify-center">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="max-w-xs md:max-w-md lg:max-w-lg bg-blue-50 rounded-xl p-3">
                    <p class="font-medium text-blue-800">보이스피싱 방지 AI</p>
                    <img src="${message}" class="mt-2 rounded-lg shadow-md max-w-full"/>
                </div>
            </div>
        `;
        chatArea.insertAdjacentHTML('beforeend', messageElement);
        scrollToBottom();
    }

    // HTML 이스케이프
    function escapeHtml(text) {
        return text.replace(/[&<>"']/g, function(m) {
            return ({
                '&': '&amp;',
                '<': '<',
                '>': '>',
                '"': '"',
                "'": '&#39;'
            })[m];
        });
    }

    // 메시지 전송
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        addUserMessage(message);
        conversationHistory.push(message);
        userInput.value = '';
        typingIndicator.classList.remove('hidden');
        scrollToBottom();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    conversation_history: conversationHistory.slice(0, -1) // 마지막은 방금 보낸 메시지
                })
            });
            if (!response.ok) throw new Error('서버 오류');
            const data = await response.json();
            addAiMessage(data.message);
            conversationHistory = data.conversation_history || [];
        } catch (error) {
            addAiMessage('서버와의 통신에 실패했습니다. 잠시 후 다시 시도해주세요.');
        } finally {
            typingIndicator.classList.add('hidden');
            scrollToBottom();
        }
    }

    // 이벤트 리스너 등록
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') sendMessage();
    });

    uploadBtn.addEventListener('click', function() {
        fileUploadOptions.classList.toggle('hidden');
    });

    closeUploadBtn.addEventListener('click', function() {
        fileUploadOptions.classList.add('hidden');
    });

    // 파일 업로드(데모)
    document.getElementById('media-upload').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            fileUploadOptions.classList.add('hidden');
            addUserMessage("동영상/이미지 파일을 업로드했습니다.");
            typingIndicator.classList.remove('hidden');
            scrollToBottom();

            const formData = new FormData(); // FormData 객체 생성

            // 선택된 파일의 0번째(e.target.files[0])를 FormData에 추가
            formData.append('image', e.target.files[0]); 

            // 서버로 파일 전송
            fetch('/api/deepfake/analyze_image', { 
                method: 'POST',
                body: formData,
            })
            .then(response => {  // 서버에서 4xx, 5xx 오류 응답 시 처리
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error || '서버 분석 중 오류 발생'); });
                }
                return response.json(); // 정상 응답은 JSON으로 파싱
            })
            .then(data => {  // 전달 받은 데이터 내용물 확인
                console.log('전체 서버 응답 (data):', JSON.stringify(data, null, 2));
                addAiMessage(JSON.stringify(data, null, 2));
            })
            .catch(error => {
                console.error(error);
                addAiMessage("파일 업로드 중 오류가 발생했습니다.");
            })
            .finally(()=>{
                e.target.value = '';  // 파일 선택 초기화 (같은 파일 다시 올릴 수 있도록)
            });
            
            // 1초 뒤에 로딩 UI 요소를 제거합니다
            // TODO: 실제로 모든 답변을 다 받았을 때 제거하도록 함
            setTimeout(() => {
                typingIndicator.classList.add('hidden');  // 로딩 UI 제거
                addAiMessage("이미지 전송 1초 지남!");
            }, 1000);
        }
    });

    document.getElementById('audio-upload').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            fileUploadOptions.classList.add('hidden');
            addUserMessage("음성 파일을 업로드했습니다.");
            typingIndicator.classList.remove('hidden');
            scrollToBottom();
            setTimeout(() => {
                typingIndicator.classList.add('hidden');
                addAiMessage("음성 파일을 분석 중입니다. (데모 버전)");
            }, 1500);
        }
    });
});