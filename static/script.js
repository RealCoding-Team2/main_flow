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

            // fetch는 자바스크립트에 POST방식으로 파일등의 데이터를 보낼 수 있게 해주는 메서드입니다
            // HTML처럼 똑같이 formData만을 보내야하며, 그렇기에 전송 전 formData형식으로 파일을 포장합니다
            // 이때 선택된 파일은 e.target.files 배열에 있으며, 일단은 0번째 요소만 포장하여 보냅니다.
            // 선택된 파일(e.target.files[0])을 FormData에 추가
            formData.append('image', e.target.files[0]); 

            // 서버로 파일 전송
            fetch('/upload', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
                // 이 부분은 정상적으로 이미지를 전송한 후 서버에서 응답(200)했을 때 콜백 함수 입니다.
                .then(data => {
                    if (data.url) {
                        addAiMessage_img(data.url);  // 서버에서 이미지를 저장한 경로를 토대로 창에 띄우기
                    }
                })
                .catch(error => {
                    console.error(error);
                    addAiMessage("파일 업로드 중 오류가 발생했습니다.");
                });
            setTimeout(() => {
                typingIndicator.classList.add('hidden');
                addAiMessage("이미지 파일을 전송함!");
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