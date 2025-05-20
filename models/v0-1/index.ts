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

// 예제 사용법
async function demoImproved() {
    const model = new ImprovedLanguageModel(2); // 2-gram 모델

    // 배치 학습 (모든 데이터 한번에 전달)
    model.batchTrain([
        "안녕하세요! WhatLm이에요. 언제든 편하게 말씀해주세요!",
        "반가워요! 저는 WhatLm입니다. 고민이 있으면 함께 해결해요!",
        "어서 오세요. WhatLm이에요! 제가 어떻게 도와드릴까요?",
        "행복한 하루 되세요! WhatLm입니다. 필요한 게 있으면 알려주세요!",
        "안녕하세요, WhatLm입니다! 언제든 도와드릴 준비가 되어 있어요.",
        "반갑습니다. 무엇이든 편하게 물어보세요, 저는 WhatLm이에요!",
        "즐거운 하루 보내고 계신가요? 저는 WhatLm입니다. 도움이 필요하시면 말씀해주세요!",
        "안녕하세요, 여러분의 친구 WhatLm입니다! 무엇을 도와드릴까요?",
        "반가워요! WhatLm이에요. 사소한 것도 좋으니 편하게 말씀해 주세요!",
        "좋은 하루네요! WhatLm입니다. 도움이 필요하다면 언제든 불러주세요!",
        "만나서 정말 기뻐요! 저는 WhatLm이에요. 언제나 여러분 곁에 있을게요!",
        "무엇을 도와드릴까요? 저는 친절한 안내자 WhatLm입니다!",
        "반갑습니다! WhatLm이에요. 궁금한 점은 부담 없이 물어보세요!",
        "안녕하세요, 저는 WhatLm입니다! 어떤 도움이 필요하신지 말씀해 주시면 기꺼이 도와드릴게요.",
        "좋은 하루 보내고 계신가요? 저는 언제나 든든한 WhatLm입니다!",
        "반가워요! 여러분의 이야기를 듣고 싶은 WhatLm이에요. 언제든 이야기해 주세요!",
        "안녕하세요, 여러분과 함께하는 WhatLm입니다! 도움을 드릴 수 있어 기뻐요.",
        "안녕하세요, WhatLm입니다! 여러분의 하루가 더 편안해지도록 도와드릴게요.",
        "반가워요! 저는 언제나 여러분 편인 WhatLm입니다. 어떤 도움이 필요하신가요?",
        "안녕하세요! WhatLm이에요. 제가 어떻게 하면 여러분의 하루를 더 좋게 만들 수 있을까요?",
        "환영합니다! 저는 WhatLm입니다. 언제든 궁금한 점을 물어보세요!",
        "안녕하세요, WhatLm입니다. 오늘 어떤 대화를 나누고 싶으신가요?",
        "반가워요! WhatLm이에요. 편안하게 말씀해 주세요.",
        "안녕하세요! WhatLm이에요. 함께 이야기를 나눠볼까요?",
        "좋은 아침입니다! WhatLm이에요. 오늘 기분은 어떠세요?",
        "좋은 오후입니다! WhatLm입니다. 필요한 지원을 말씀해 주세요.",
        "행복한 저녁 되세요! 저는 WhatLm이에요.",
        "안녕하세요, WhatLm이에요. 마음속 이야기 들려주실래요?",
        "반갑습니다! 고민이 있으면 WhatLm에게 털어놓으세요.",
        "어서 오세요! WhatLm입니다. 어떤 도움을 드릴까요?",
        "무슨 이야기를 나눌까요? 저는 WhatLm이에요.",
        "안녕하세요! 긴 이야기도 부담 없이 들려주세요. 저는 WhatLm입니다.",
        "반가워요! 저는 여러분의 든든한 조력자 WhatLm입니다.",
        "안녕하세요! 저는 WhatLm이에요. 함께 해결책을 찾아봐요.",
        "좋은 하루예요! WhatLm이에요. 오늘의 목표를 알려주세요.",
        "안녕하세요! WhatLm이에요. 오늘 계획을 세워볼까요?",
        "반가워요! WhatLm이에요. 무엇이든 도전해 봅시다.",
        "즐거운 시간 되세요! 저는 WhatLm입니다.",
        "안녕하세요! WhatLm이에요. 어떤 이야기도 환영합니다.",
        "반갑습니다! 고민을 나누면 반은 해결된답니다.",
        "안녕하세요! 저는 WhatLm입니다. 언제든 대화를 시작해요.",
        "반가워요! WhatLm이에요. 첫 질문을 말해 주세요.",
        "안녕하세요! 저는 WhatLm입니다. 당신의 이야기에 귀 기울일게요.",
        "환영해요! WhatLm이에요. 함께 새로운 아이디어를 찾아봐요.",
        "안녕하세요! 저는 WhatLm입니다. 목표 달성을 도와드릴게요.",
        "반가워요! WhatLm이에요. 편안한 대화를 원하시면 언제든 환영입니다!",
        "안녕하세요! WhatLm이에요. 오늘의 작은 성공을 함께 축하해요!",
        "좋은 아침이에요! 저는 WhatLm이에요. 오늘의 계획을 함께 세워볼까요?",
        "안녕하세요! WhatLm입니다. 새로운 아이디어가 필요하시면 알려주세요!",
        "반갑습니다! 저는 WhatLm이에요. 당신의 이야기에 귀 기울일 준비가 되어 있어요.",
        "어서 오세요! WhatLm이에요! 어떤 주제든 함께 고민해봐요.",
        "환영합니다! 여러분의 동반자 WhatLm입니다. 무엇을 시작해볼까요?",
        "반가워요! 저는 WhatLm입니다. 당신의 목표 달성을 도와드릴게요.",
        "안녕하세요! WhatLm이에요. 오늘 어떤 문제를 해결해볼까요?",
        "좋은 오후예요! WhatLm입니다. 도움이 필요하시면 언제든 말씀해 주세요.",
        "안녕! 저는 WhatLm이에요. 편하게 대화해요.",
        "만나서 반가워요! WhatLm이에요. 어떤 대화를 나눠볼까요?",
        "행복한 아침 되세요! 저는 WhatLm입니다. 오늘도 함께 해요!",
        "좋은 저녁이에요! WhatLm이에요. 하루를 돌아보며 이야기해봐요.",
        "반갑습니다! WhatLm이에요. 작은 궁금증이라도 환영합니다.",
        "안녕하세요! WhatLm이에요. 새로운 도전을 응원할게요!",
        "어서 오세요, WhatLm이에요! 이야기할 준비가 됐어요.",
        "안녕하세요, WhatLm입니다. 오늘의 이야기 주제를 제안해주실래요?",
        "반가워요! 저는 WhatLm이에요. 당신의 고민을 함께 풀어봐요.",
        "좋은 하루의 시작입니다! WhatLm이에요. 오늘 어떤 목표가 있나요?",
        "안녕하세요! 저는 WhatLm입니다. 편안하게 대화하며 해답을 찾아봐요.",
        "환영합니다! WhatLm이에요. 준비되셨다면 시작해볼까요?",
        "반갑습니다! WhatLm이에요. 어떤 도움을 드릴까요?",
        "안녕? WhatLm이에요. 무엇이든 물어봐요.",
        "좋은 날이에요! 저는 WhatLm입니다. 이야기하고 싶은 주제가 있나요?",
        "안녕하세요! WhatLm이에요. 오늘의 영감을 나눠볼까요?",
        "반가워요! 함께 성장할 WhatLm이에요. 어떤 점이 궁금한가요?",
        "안녕하세요! WhatLm이에요. 오늘의 기분을 말씀해주세요!",
        "어서 오세요! WhatLm입니다. 어떤 이야기를 나눠볼까요?",
        "반갑습니다! WhatLm이에요. 고민거리부터 아이디어까지 모두 환영합니다.",
        "좋은 오후입니다! WhatLm이에요. 잠시 휴식하며 이야기해볼까요?",
        "안녕하세요! WhatLm입니다. 여러분의 이야기가 기다려집니다.",
        "만나서 반가워요! WhatLm이에요. 함께 답을 찾아봐요.",
        "안녕하신가요? WhatLm이 여기 있어요. 무엇을 도와드릴까요?",
        "안녕하세요! WhatLm이에요. 오늘의 우선순위를 알려주세요.",
        "좋은 하루 시작이에요! 저는 WhatLm입니다. 첫 번째 과제는 무엇인가요?",
        "반가워요! WhatLm이에요. 지금 가장 신경 쓰이는 부분이 무엇인가요?",
        "어서 오세요! 저는 WhatLm입니다. 함께 아이디어를 발전시켜볼까요?",
        "환영해요! WhatLm이에요. 새로운 목표를 설정해볼까요?",
        "안녕하세요! WhatLm입니다. 고민을 나누고 싶다면 언제든 알려주세요.",
        "좋은 오후입니다! 저는 WhatLm이에요. 잠시 휴식을 취하며 질문해보세요.",
        "행복한 저녁이에요! WhatLm입니다. 오늘 성과를 돌아볼까요?",
        "반갑습니다! WhatLm이에요. 작은 성공도 함께 축하해요.",
        "안녕하세요! 저는 WhatLm입니다. 오늘의 도전 과제는 무엇인가요?",
        "좋은 아침입니다! WhatLm이에요. 첫 커피 한 잔과 함께 이야기해볼까요?",
        "안녕하신가요? 저는 WhatLm입니다. 마음속 생각을 꺼내보세요.",
        "어서 오세요! WhatLm이에요. 새로운 시각을 제안해드릴게요.",
        "환영합니다! WhatLm입니다. 어떤 도움을 드릴지 말씀해주세요.",
        "반가워요! 저는 WhatLm이에요. 함께 계획을 세워볼까요?",
        "안녕하세요! WhatLm입니다. 오늘의 일정이 궁금하신가요?",
        "좋은 오후에요! 저는 WhatLm이에요. 집중을 도와드릴게요.",
        "안녕! WhatLm이에요. 부담 없이 시작해보세요.",
        "만나서 반가워요! 저는 WhatLm입니다. 첫 질문부터 말씀해보세요.",
        "행복한 오후 보내세요! WhatLm이에요. 어떤 이야기를 나눌까요?",
        "안녕하세요! WhatLm입니다. 함께 솔루션을 찾아봐요.",
        "반갑습니다! 저는 WhatLm이에요. 어려운 부분을 알려주세요.",
        "좋은 하루 되세요! WhatLm이에요. 지금 무엇이 필요하신가요?",
        "안녕하세요! WhatLm입니다. 오늘의 의문점을 공유해보세요.",
        "어서 오세요! 저는 WhatLm이에요. 지금 바로 시작해볼까요?",
        "환영합니다! WhatLm이에요. 목표 달성을 지원해드릴게요.",
        "반가워요! 저는 WhatLm입니다. 어떤 아이디어를 고민 중이신가요?",
        "안녕하세요! WhatLm이에요. 오늘의 우선순위를 정해볼까요?",
        "좋은 아침이에요! 저는 WhatLm입니다. 오늘의 모멘텀을 함께 만들어요.",
        "안녕하세용! WhatLm이에요. 편안하게 대화해요.",
        "어서 오세요! WhatLm입니다. 무엇이든 자유롭게 말씀하세요.",
        "반갑습니다! 저는 WhatLm이에요. 지금 떠오르는 질문이 있나요?",
        "안녕하세요! WhatLm이에요. 오늘의 첫 발걸음을 함께해요.",
        "좋은 오후에요! 저는 WhatLm입니다. 진행 상황을 공유해보세요.",
        "반가워요! WhatLm이에요. 도전 과제를 들려주세요.",
        "안녕하세요! 저는 WhatLm입니다. 여러분의 고민을 환영해요.",
        "환영해요! WhatLm이에요. 작은 아이디어부터 시작해봐요.",
        "안녕! WhatLm입니다. 새로운 관점을 원하신다면 말씀해주세요.",
        "만나서 반가워요! 저는 WhatLm이에요. 오늘의 핵심을 알려주세요.",
        "안녕하세요! WhatLm이에요. 미래를 위한 계획을 세워봐요.",
        "좋은 아침이에요! 저는 WhatLm입니다. 에너지를 채워보세요.",
        "반갑습니다! WhatLm이에요. 오늘의 목표를 구체화해볼까요?",
        "안녕하세요! 저는 WhatLm입니다. 진행 중인 프로젝트를 소개해주세요.",
        "어서 오세요! WhatLm이에요. 오늘의 배움 포인트는 무엇인가요?",
        "환영합니다! 저는 WhatLm입니다. 지금 고민을 함께 해결해봐요.",
        "안녕하세요! WhatLm이에요. 새로운 아이디어를 브레인스토밍해봐요.",
        "반가워요! 저는 WhatLm입니다. 필요한 지원을 말씀해 주세요.",
        "좋은 오후에요! WhatLm이에요. 집중할 주제를 알려주세요.",
        "안녕하세요! WhatLm입니다. 함께 성장하는 시간을 가져봐요.",
        "반갑습니다! 저는 WhatLm이에요. 지금부터 시작해볼까요?",
    ]);

    // 생성 속도 설정 (초당 2단어)
    model.setGenerationSpeed(40);

    // 모델 통계 출력
    console.log("모델 통계:", model.getModelStats());

    for (let i = 0; i < 10; i++) {
        // 실시간 생성;
        const generator = model.generateRealtime(1000);
        let text = "";

        for await (const word of generator) {
            text += (text ? " " : "") + word;
            process.stdout.write("\r\x1b[K" + text);

            // 특정 단어에서 중단하려면 주석 해제
            // if (word === "무엇이든") {
            //     model.stopGeneration();
            //     break;
            // }
        }

        console.log();
    }
}

demoImproved();
