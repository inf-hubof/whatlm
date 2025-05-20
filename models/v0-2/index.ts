export class ImprovedLanguageModel {
    // 트랜지션 확률 테이블 (단어 전이 확률)
    private transitions: Map<string, Map<string, number>> = new Map();
    // 확률 계산이 완료된 캐시 (성능 향상을 위해)
    private normalizedTransitions: Map<string, Map<string, number>> = new Map();
    // 시작 단어 및 빈도
    private startWordFrequencies: Map<string, number> = new Map();

    private isGenerating: boolean = false;
    private generationDelay: number = 1000;
    private contextSize: number = 2; // N-gram 크기 (기본값: 2)

    constructor(contextSize: number = 2) {
        this.contextSize = Math.max(1, contextSize);
    }

    train(texts: string | string[]): void {
        // 배열이 아니면 배열로 변환하여 배치 처리
        const textArray = Array.isArray(texts) ? texts : [texts];

        // 각 텍스트에 대해 학습 수행
        for (const text of textArray) {
            const words = text.split(/\s+/).filter((word) => word.length > 0);
            if (words.length === 0) continue;

            // 시작 단어 빈도 업데이트
            const startWord = words[0];
            this.startWordFrequencies.set(
                startWord,
                (this.startWordFrequencies.get(startWord) || 0) + 1
            );

            // N-gram 기반 트랜지션 구축
            if (this.contextSize === 1) {
                // 기존 방식과 동일 (bigram)
                for (let i = 0; i < words.length - 1; i++) {
                    this.addTransition(words[i], words[i + 1]);
                }
            } else {
                // N-gram 처리 (contextSize > 1)
                for (let i = 0; i < words.length - 1; i++) {
                    // 현재 위치에서 가능한 최대 컨텍스트 크기 계산
                    const contextSize = Math.min(this.contextSize, i + 1);
                    // 현재 위치에서 이전 contextSize 단어를 포함한 컨텍스트 생성
                    const context = words
                        .slice(i - contextSize + 1, i + 1)
                        .join(" ");
                    this.addTransition(context, words[i + 1]);
                }
            }
        }

        // 정규화된 전이 확률 캐시 무효화 (학습 후 필요할 때 다시 계산)
        this.normalizedTransitions.clear();
    }

    private addTransition(current: string, next: string): void {
        if (!this.transitions.has(current)) {
            this.transitions.set(current, new Map());
        }

        const nextWords = this.transitions.get(current)!;
        nextWords.set(next, (nextWords.get(next) || 0) + 1);
    }

    // 확률 테이블 정규화 (캐싱하여 성능 개선)
    private getNormalizedTransitions(
        currentContext: string
    ): Map<string, number> {
        // 이미 계산된 정규화 확률이 있으면 반환
        if (this.normalizedTransitions.has(currentContext)) {
            return this.normalizedTransitions.get(currentContext)!;
        }

        // 현재 컨텍스트에 대한 전이가 없는 경우 빈 맵 반환
        if (!this.transitions.has(currentContext)) {
            return new Map();
        }

        const transitions = this.transitions.get(currentContext)!;
        const normalized = new Map<string, number>();

        // 총 빈도수 계산
        const total = Array.from(transitions.values()).reduce(
            (sum, count) => sum + count,
            0
        );

        // 각 전이 확률 정규화
        for (const [word, count] of transitions.entries()) {
            normalized.set(word, count / total);
        }

        // 계산된 확률 캐싱
        this.normalizedTransitions.set(currentContext, normalized);

        return normalized;
    }

    // 가중치 기반 다음 단어 선택 (최적화)
    private getNextWord(currentContext: string): string | null {
        const normalizedTransitions =
            this.getNormalizedTransitions(currentContext);
        if (normalizedTransitions.size === 0) return null;

        // 누적 분포 함수(CDF) 방식으로 샘플링
        const random = Math.random();
        let cumulativeProbability = 0;

        for (const [word, probability] of normalizedTransitions.entries()) {
            cumulativeProbability += probability;
            if (random <= cumulativeProbability) {
                return word;
            }
        }

        // 혹시 모를 경우를 대비한 기본값
        return Array.from(normalizedTransitions.keys())[0];
    }

    // 시작 단어 선택 (빈도 기반)
    private getStartWord(): string {
        const words: string[] = [];
        const weights: number[] = [];

        for (const [word, count] of this.startWordFrequencies.entries()) {
            words.push(word);
            weights.push(count);
        }

        // 가중치 기반 랜덤 선택
        const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
        let random = Math.random() * totalWeight;

        for (let i = 0; i < words.length; i++) {
            random -= weights[i];
            if (random <= 0) return words[i];
        }

        return words[0];
    }

    // 비동기 실시간 생성
    async *generateRealtime(maxLength: number = 50) {
        if (this.startWordFrequencies.size === 0) return;

        this.isGenerating = true;
        const currentWord = this.getStartWord();
        let context = currentWord;

        yield currentWord;

        let wordCount = 1;
        while (wordCount < maxLength && this.isGenerating) {
            const nextWord = this.getNextWord(context);

            if (nextWord === null) break;

            await new Promise((resolve) =>
                setTimeout(resolve, this.generationDelay)
            );

            yield nextWord;

            // N-gram 컨텍스트 업데이트
            if (this.contextSize > 1) {
                const contextWords = context.split(" ");
                // 컨텍스트 윈도우 이동
                if (contextWords.length >= this.contextSize) {
                    contextWords.shift(); // 가장 오래된 단어 제거
                }
                contextWords.push(nextWord); // 새 단어 추가
                context = contextWords.join(" ");
            } else {
                context = nextWord;
            }

            wordCount++;
        }
    }

    // 일반 텍스트 생성
    generate(maxLength: number = 50): string {
        if (this.startWordFrequencies.size === 0) return "";

        const currentWord = this.getStartWord();
        let context = currentWord;

        const result: string[] = [currentWord];

        for (let i = 1; i < maxLength; i++) {
            const nextWord = this.getNextWord(context);
            if (nextWord === null) break;

            result.push(nextWord);

            // N-gram 컨텍스트 업데이트
            if (this.contextSize > 1) {
                const contextWords = context.split(" ");
                if (contextWords.length >= this.contextSize) {
                    contextWords.shift();
                }
                contextWords.push(nextWord);
                context = contextWords.join(" ");
            } else {
                context = nextWord;
            }
        }

        return result.join(" ");
    }

    stopGeneration(): void {
        this.isGenerating = false;
    }

    setGenerationSpeed(wordsPerSecond: number): void {
        this.generationDelay = 1000 / wordsPerSecond;
    }

    // 배치 학습을 위한 유틸리티 메서드
    batchTrain(texts: string[]): void {
        this.train(texts);
    }

    // 모델 통계 정보
    getModelStats() {
        return {
            uniqueWords: this.getUniqueWordCount(),
            uniqueContexts: this.transitions.size,
            startWords: this.startWordFrequencies.size,
            memoryUsage: this.estimateMemoryUsage(),
        };
    }

    private getUniqueWordCount(): number {
        const uniqueWords = new Set<string>();

        // 모든 다음 단어 수집
        for (const nextWords of this.transitions.values()) {
            for (const word of nextWords.keys()) {
                uniqueWords.add(word);
            }
        }

        // 모든 컨텍스트 단어 수집
        for (const context of this.transitions.keys()) {
            for (const word of context.split(" ")) {
                uniqueWords.add(word);
            }
        }

        return uniqueWords.size;
    }

    private estimateMemoryUsage(): string {
        let count = 0;

        // 트랜지션 맵 크기 추정
        for (const [context, nextWords] of this.transitions.entries()) {
            count += context.length * 2; // 문자열 크기 (2바이트 가정)
            count += 8; // Map 엔트리 오버헤드

            for (const [word] of nextWords.entries()) {
                count += word.length * 2; // 문자열 크기
                count += 12; // Map 엔트리 + 숫자 크기
            }
        }

        // 시작 단어 맵 크기 추정
        for (const [word] of this.startWordFrequencies.entries()) {
            count += word.length * 2;
            count += 12;
        }

        // 정규화된 트랜지션 맵 크기 추정
        for (const [context, probs] of this.normalizedTransitions.entries()) {
            count += context.length * 2;
            count += 8;

            for (const [word] of probs.entries()) {
                count += word.length * 2;
                count += 12;
            }
        }

        // 메모리 크기를 적절한 단위로 변환
        if (count < 1024) {
            return `${count} bytes`;
        } else if (count < 1024 * 1024) {
            return `${(count / 1024).toFixed(2)} KB`;
        } else {
            return `${(count / (1024 * 1024)).toFixed(2)} MB`;
        }
    }
}

// 지식 데이터 타입 정의
type KnowledgeData =
    | string
    | string[]
    | Record<string, string | string[] | number | boolean>;

export class QALanguageModel extends ImprovedLanguageModel {
    // Q&A 데이터베이스: 질문과 정답 쌍 저장
    private qaDatabase: Map<string, string[]> = new Map();
    // 유사 질문 검색을 위한 인덱스
    private questionIndex: Map<string, Set<string>> = new Map();
    // 추가된 임베딩 유사도 비교를 위한 단어 벡터 (간단한 구현용)
    private wordVectors: Map<string, number[]> = new Map();
    // 상황별 응답 템플릿
    private responseTemplates: Map<string, string[]> = new Map();
    // 학습 완료된 질문 목록 (중복 방지용)
    private trainedQuestions: Set<string> = new Set();
    // 질문 응답 시 참조할 수 있는 지식 베이스
    private knowledgeBase: Map<string, KnowledgeData> = new Map();

    // 지식 데이터 타입 지정
    constructor(contextSize: number = 2) {
        super(contextSize);
        this.initializeResponseTemplates();
        this.initializeWordVectors();
    }

    // 응답 템플릿 초기화
    private initializeResponseTemplates() {
        this.responseTemplates.set("not_found", [
            "죄송합니다, 해당 질문에 대한 답변을 찾지 못했어요. 다른 방식으로 질문해 주시겠어요?",
            "정확한 답변을 드리기 어렵네요. 조금 더 구체적인 질문을 해주시겠어요?",
            "아직 그 질문에 대한 정보가 부족합니다. 더 많은 정보를 제공해주시면 도움이 될 것 같아요.",
        ]);
    }

    // 단어 벡터 초기화 (간단한 구현)
    private initializeWordVectors() {
        // 실제 구현에서는 사전 훈련된 임베딩 모델을 사용하는 것이 좋습니다.
        // 여기서는 예시로 간단한 랜덤 벡터를 생성합니다.
        const commonWords = [
            "안녕",
            "무엇",
            "어떻게",
            "왜",
            "언제",
            "어디",
            "누구",
            "질문",
            "답변",
            "도움",
            "알려줘",
            "정보",
            "방법",
            "문제",
            "해결",
            "설명",
            "이해",
            "예제",
            "의미",
        ];

        for (const word of commonWords) {
            // 간단한 5차원 임베딩 벡터 생성 (실제로는 더 많은 차원을 사용)
            const vector = Array(5)
                .fill(0)
                .map(() => Math.random() - 0.5);
            this.wordVectors.set(word, this.normalizeVector(vector));
        }
    }

    // 벡터 정규화
    private normalizeVector(vector: number[]): number[] {
        const magnitude = Math.sqrt(
            vector.reduce((sum, val) => sum + val * val, 0)
        );
        return vector.map((val) => val / magnitude);
    }

    // Q&A 데이터 학습
    trainQA(question: string, answers: string | string[]): void {
        // 질문 전처리 (공백 정규화, 소문자 변환 등)
        const normalizedQuestion = this.normalizeText(question);

        if (this.trainedQuestions.has(normalizedQuestion)) {
            // 이미 학습된 질문이면 답변 추가
            const existingAnswers =
                this.qaDatabase.get(normalizedQuestion) || [];
            const answerArray = Array.isArray(answers) ? answers : [answers];

            // 중복 답변 방지
            for (const answer of answerArray) {
                if (!existingAnswers.includes(answer)) {
                    existingAnswers.push(answer);
                }
            }

            this.qaDatabase.set(normalizedQuestion, existingAnswers);
        } else {
            // 새로운 질문 추가
            this.trainedQuestions.add(normalizedQuestion);
            this.qaDatabase.set(
                normalizedQuestion,
                Array.isArray(answers) ? answers : [answers]
            );

            // 인덱싱을 위해 질문 토큰화 및 저장
            const tokens = this.tokenize(normalizedQuestion);
            for (const token of tokens) {
                if (!this.questionIndex.has(token)) {
                    this.questionIndex.set(token, new Set());
                }
                this.questionIndex.get(token)!.add(normalizedQuestion);
            }

            // 기존 언어 모델에도 학습하여 자연스러운 문장 생성에 활용
            super.train(answers);
        }
    }

    // 지식 베이스에 정보 추가
    addKnowledge(key: string, value: KnowledgeData): void {
        this.knowledgeBase.set(key, value);
    }

    // 여러 Q&A 쌍 학습
    batchTrainQA(
        qaPairs: Array<{
            question: string | string[];
            answers: string | string[];
        }>
    ): void {
        for (const { question, answers } of qaPairs) {
            if (Array.isArray(question)) {
                // 여러 질문에 대해 동일한 답변을 학습
                for (const q of question) {
                    this.trainQA(q, answers);
                }
            } else {
                // 기존 방식과 동일
                this.trainQA(question, answers);
            }
        }
    }

    // 텍스트 정규화
    private normalizeText(text: string): string {
        return text.trim().toLowerCase();
    }

    // 텍스트 토큰화 (간단한 공백 기반 토큰화)
    private tokenize(text: string): string[] {
        // 특수문자 제거 및 공백으로 분리
        return text
            .replace(/[^\w\s가-힣]/g, "")
            .split(/\s+/)
            .filter((token) => token.length > 0);
    }

    // 질문 유사도 계산
    private calculateSimilarity(question1: string, question2: string): number {
        // 1. 토큰 일치도 계산
        const tokens1 = new Set(this.tokenize(question1));
        const tokens2 = new Set(this.tokenize(question2));

        // 교집합 크기
        const intersection = new Set(
            [...tokens1].filter((token) => tokens2.has(token))
        );

        // 자카드 유사도
        const union = new Set([...tokens1, ...tokens2]);
        const jaccardSim = intersection.size / union.size;

        // 2. 단어 벡터 유사도 (코사인 유사도)
        let vectorSim = 0;
        if (tokens1.size > 0 && tokens2.size > 0) {
            // 질문 벡터 계산 (단어 벡터의 평균)
            const vector1 = this.calculateQuestionVector(question1);
            const vector2 = this.calculateQuestionVector(question2);

            if (vector1 && vector2) {
                // 코사인 유사도 계산
                vectorSim = this.calculateCosineSimilarity(vector1, vector2);
            }
        }

        // 3. 편집 거리 기반 유사도
        const editSim =
            1 -
            this.calculateEditDistance(question1, question2) /
                Math.max(question1.length, question2.length);

        // 가중 평균으로 최종 유사도 계산
        return jaccardSim * 0.4 + vectorSim * 0.4 + editSim * 0.2;
    }

    // 질문 벡터 계산
    private calculateQuestionVector(question: string): number[] | null {
        const tokens = this.tokenize(question);
        const vectors: number[][] = [];

        for (const token of tokens) {
            if (this.wordVectors.has(token)) {
                vectors.push(this.wordVectors.get(token)!);
            }
        }

        if (vectors.length === 0) return null;

        // 각 차원별 평균 계산
        const dimensions = vectors[0].length;
        const avgVector = Array(dimensions).fill(0);

        for (const vector of vectors) {
            for (let i = 0; i < dimensions; i++) {
                avgVector[i] += vector[i] / vectors.length;
            }
        }

        return this.normalizeVector(avgVector);
    }

    // 코사인 유사도 계산
    private calculateCosineSimilarity(
        vector1: number[],
        vector2: number[]
    ): number {
        if (vector1.length !== vector2.length) return 0;

        let dotProduct = 0;
        let magnitude1 = 0;
        let magnitude2 = 0;

        for (let i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
            magnitude1 += vector1[i] * vector1[i];
            magnitude2 += vector2[i] * vector2[i];
        }

        magnitude1 = Math.sqrt(magnitude1);
        magnitude2 = Math.sqrt(magnitude2);

        if (magnitude1 === 0 || magnitude2 === 0) return 0;

        return dotProduct / (magnitude1 * magnitude2);
    }

    // 편집 거리 계산 (Levenshtein 거리)
    private calculateEditDistance(str1: string, str2: string): number {
        const m = str1.length;
        const n = str2.length;

        // 동적 프로그래밍 테이블 초기화
        const dp: number[][] = Array(m + 1)
            .fill(null)
            .map(() => Array(n + 1).fill(0));

        // 기본 케이스 초기화
        for (let i = 0; i <= m; i++) dp[i][0] = i;
        for (let j = 0; j <= n; j++) dp[0][j] = j;

        // 편집 거리 계산
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (str1[i - 1] === str2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] =
                        1 +
                        Math.min(
                            dp[i - 1][j], // 삭제
                            dp[i][j - 1], // 삽입
                            dp[i - 1][j - 1] // 대체
                        );
                }
            }
        }

        return dp[m][n];
    }

    // 가장 유사한 질문 찾기
    private findSimilarQuestions(
        question: string,
        threshold: number = 0.7
    ): string[] {
        const normalizedQuestion = this.normalizeText(question);
        const tokens = this.tokenize(normalizedQuestion);

        // 후보 질문 집합 구성
        const candidateQuestions = new Set<string>();

        // 토큰 기반 후보 수집
        for (const token of tokens) {
            if (this.questionIndex.has(token)) {
                for (const q of this.questionIndex.get(token)!) {
                    candidateQuestions.add(q);
                }
            }
        }

        // 모든 질문에서 최소한의 후보 확보
        if (candidateQuestions.size < 5) {
            for (const q of this.qaDatabase.keys()) {
                candidateQuestions.add(q);
                if (candidateQuestions.size >= 10) break;
            }
        }

        // 유사도 계산 및 정렬
        const similarQuestions: [string, number][] = [];

        for (const candidateQuestion of candidateQuestions) {
            const similarity = this.calculateSimilarity(
                normalizedQuestion,
                candidateQuestion
            );

            if (similarity >= threshold) {
                similarQuestions.push([candidateQuestion, similarity]);
            }
        }

        // 유사도로 정렬
        similarQuestions.sort((a, b) => b[1] - a[1]);

        // 가장 유사한 질문들 반환
        return similarQuestions.map(([question]) => question);
    }

    // 질문에 대한 답변 생성
    async answerQuestion(question: string): Promise<string> {
        const normalizedQuestion = this.normalizeText(question);

        // 1. 정확히 일치하는 질문 확인
        if (this.qaDatabase.has(normalizedQuestion)) {
            const answers = this.qaDatabase.get(normalizedQuestion)!;
            // 여러 답변 중 랜덤하게 선택
            return answers[Math.floor(Math.random() * answers.length)];
        }

        // 2. 유사 질문 검색
        const similarQuestions = this.findSimilarQuestions(normalizedQuestion);

        if (similarQuestions.length > 0) {
            // 가장 유사한 질문에 대한 답변 선택
            const bestMatch = similarQuestions[0];
            const answers = this.qaDatabase.get(bestMatch)!;

            return answers[Math.floor(Math.random() * answers.length)];
        }

        // 3. 지식 베이스 검색
        const relevantInfo = this.searchKnowledgeBase(normalizedQuestion);
        if (relevantInfo) {
            return this.generateAnswerFromKnowledge(relevantInfo);
        }

        // 4. 유사 질문이 없는 경우, 기본 응답 생성
        return this.generateDefaultResponse();
    }

    // 지식 베이스 검색
    private searchKnowledgeBase(question: string): KnowledgeData | null {
        const tokens = this.tokenize(question);

        // 간단한 키워드 매칭 기반 검색
        for (const [key, value] of this.knowledgeBase.entries()) {
            const keyTokens = this.tokenize(key);
            const matches = keyTokens.filter((token) => tokens.includes(token));

            // 50% 이상 일치하면 관련 정보로 판단
            if (matches.length >= keyTokens.length * 0.5) {
                return value;
            }
        }

        return null;
    }

    // 지식 베이스 기반 답변 생성
    private generateAnswerFromKnowledge(knowledge: KnowledgeData): string {
        // 지식 데이터 기반 응답 생성
        if (typeof knowledge === "string") {
            return knowledge;
        } else if (Array.isArray(knowledge)) {
            return knowledge.join(", ");
        }

        return this.generateDefaultResponse();
    }

    // 기본 응답 생성
    private generateDefaultResponse(): string {
        // 응답을 찾지 못한 경우
        const templates = this.responseTemplates.get("not_found")!;
        return templates[Math.floor(Math.random() * templates.length)];
    }

    // 대화 컨텍스트 관리를 위한 클래스
    public beginConversation(): Conversation {
        return new Conversation(this);
    }
}

// 대화 컨텍스트를 위한 타입 정의
type ContextValue = string | number | boolean | null | Array<string | number>;
type MessageType = { role: "user" | "assistant"; content: string };

// 대화 컨텍스트 관리를 위한 클래스
class Conversation {
    private qaModel: QALanguageModel;
    private messageHistory: MessageType[] = [];
    private context: Map<string, ContextValue> = new Map();

    constructor(qaModel: QALanguageModel) {
        this.qaModel = qaModel;
    }

    // 사용자 메시지 추가
    async addUserMessage(message: string): Promise<string> {
        this.messageHistory.push({ role: "user", content: message });

        // 질문 분석 및 응답 생성
        const response = await this.generateResponse(message);

        this.messageHistory.push({ role: "assistant", content: response });
        return response;
    }

    // 컨텍스트 설정
    setContext(key: string, value: ContextValue): void {
        this.context.set(key, value);
    }

    // 컨텍스트 가져오기
    getContext(key: string): ContextValue | undefined {
        return this.context.get(key);
    }

    // 이전 메시지 가져오기
    getMessageHistory(): MessageType[] {
        return [...this.messageHistory];
    }

    // 대화 맥락을 고려한 응답 생성
    private async generateResponse(userMessage: string): Promise<string> {
        // QA 모델로 응답 생성
        return await this.qaModel.answerQuestion(userMessage);
    }
}

// // 예제 사용법
// async function demoQAModel() {
//     // QA 모델 생성 (2-gram 기반)
//     const qaModel = new QALanguageModel(2);

//     // 기본 질문-답변 쌍 학습
//     qaModel.batchTrainQA([
//         {
//             question: [],
//             answers: [],
//         },
//     ]);

//     // 지식 베이스 추가
//     qaModel.addKnowledge("", "");

//     // 대화 시작
//     const conversation = qaModel.beginConversation();

//     // 대화 예시
//     const questions = [""];

//     for (const question of questions) {
//         console.log(`사용자: ${question}`);
//         const answer = await conversation.addUserMessage(question);
//         console.log(`WhatLm: ${answer}`);
//         console.log();

//         // 실제 사용 시 비동기 지연
//         await new Promise((resolve) => setTimeout(resolve, 1000));
//     }
// }

// // 모델 실행
// demoQAModel();
