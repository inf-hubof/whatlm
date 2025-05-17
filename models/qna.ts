export class QnALanguageModel {
    private qaDatabase: Map<string, string[]> = new Map();

    private transitions: Map<string, Map<string, number>> = new Map();
    private startWords: string[] = [];

    private isGenerating: boolean = false;
    private generationDelay: number = 40;

    private similarityThreshold: number = 0.3;

    trainMarkov(text: string): void {
        const words = text.split(/\s+/).filter((word) => word.length > 0);

        if (words.length === 0) return;

        this.startWords.push(words[0]);

        for (let i = 0; i < words.length - 1; i++) {
            const currentWord = words[i];
            const nextWord = words[i + 1];

            if (!this.transitions.has(currentWord))
                this.transitions.set(currentWord, new Map());

            const nextWords = this.transitions.get(currentWord)!;
            nextWords.set(nextWord, (nextWords.get(nextWord) || 0) + 1);
        }
    }

    trainQnA(question: string, answer: string): void {
        const normalizedQuestion = this.normalizeText(question);
        const keywords = this.extractKeywords(normalizedQuestion);
        const keys = this.generateKeys(keywords);

        for (const key of keys) {
            if (!this.qaDatabase.has(key)) {
                this.qaDatabase.set(key, []);
            }

            const answers = this.qaDatabase.get(key)!;
            if (!answers.includes(answer)) {
                answers.push(answer);
            }
        }

        this.trainMarkov(answer);
    }

    async answer(question: string): Promise<string> {
        const normalizedQuestion = this.normalizeText(question);
        const keywords = this.extractKeywords(normalizedQuestion);

        const bestMatch = await this.findBestMatch(
            keywords,
            normalizedQuestion
        );

        if (bestMatch) {
            return bestMatch;
        }

        return this.generate(20);
    }

    async *answerRealtime(
        question: string,
        maxLength: number = 50
    ): AsyncGenerator<string> {
        const normalizedQuestion = this.normalizeText(question);

        const keywords = this.extractKeywords(normalizedQuestion);

        const bestMatch = await this.findBestMatch(
            keywords,
            normalizedQuestion
        );

        this.isGenerating = true;

        if (bestMatch) {
            const words = bestMatch
                .split(/\s+/)
                .filter((word) => word.length > 0);

            for (const word of words) {
                if (!this.isGenerating) break;

                await new Promise((resolve) =>
                    setTimeout(resolve, this.generationDelay)
                );
                yield word;
            }
        } else {
            const generator = this.generateRealtime(maxLength);
            for await (const word of generator) {
                yield word;
            }
        }
    }

    private extractKeywords(text: string): string[] {
        const stopwords = [
            "입니다",
            "이에요",
            "합니다",
            "있나요",
            "있을까요",
            "무엇을",
            "어떤",
            "저는",
            "제",
            "전",
            "이라고",
        ];

        return text
            .split(/\s+/)
            .filter((word) => word.length > 1 && !stopwords.includes(word))
            .map((word) => word.replace(/[?.!,]/g, ""));
    }

    private normalizeText(text: string): string {
        return text.trim().toLowerCase().replace(/\s+/g, " ");
    }

    private generateKeys(keywords: string[]): string[] {
        const keys: string[] = [];

        for (const keyword of keywords) {
            keys.push(keyword);
        }

        for (let i = 0; i < keywords.length; i++) {
            for (let j = i + 1; j < keywords.length; j++) {
                keys.push(`${keywords[i]}_${keywords[j]}`);
            }
        }

        if (keywords.length > 2) {
            keys.push(keywords.join("_"));
        }

        return keys;
    }

    private async findBestMatch(
        keywords: string[],
        question: string
    ): Promise<string | null> {
        let bestScore = -1;
        let bestAnswer: string | null = null;

        const keys = this.generateKeys(keywords);
        const candidateAnswers: string[] = [];

        for (const key of keys) {
            if (this.qaDatabase.has(key)) {
                const answers = this.qaDatabase.get(key)!;
                candidateAnswers.push(...answers);
            }
        }

        const uniqueAnswers = [...new Set(candidateAnswers)];

        for (const answer of uniqueAnswers) {
            const score = this.calculateSimilarity(question, answer);

            if (score > bestScore && score >= this.similarityThreshold) {
                bestScore = score;
                bestAnswer = answer;
            }
        }

        return bestAnswer;
    }

    private calculateSimilarity(text1: string, text2: string): number {
        const set1 = new Set(text1.split(/\s+/));
        const set2 = new Set(text2.split(/\s+/));

        const intersection = new Set([...set1].filter((x) => set2.has(x)));
        const union = new Set([...set1, ...set2]);
        return intersection.size / union.size;
    }

    async *generateRealtime(maxLength: number = 50): AsyncGenerator<string> {
        if (this.startWords.length === 0) return;

        this.isGenerating = true;

        const randomStartIndex = Math.floor(
            Math.random() * this.startWords.length
        );
        let currentWord = this.startWords[randomStartIndex];

        yield currentWord;

        let wordCount = 1;
        while (wordCount < maxLength && this.isGenerating) {
            const nextWord = this.getNextWord(currentWord);

            if (nextWord === null) break;

            await new Promise((resolve) =>
                setTimeout(resolve, this.generationDelay)
            );

            yield nextWord;
            currentWord = nextWord;
            wordCount++;
        }
    }

    stopGeneration(): void {
        this.isGenerating = false;
    }

    setGenerationSpeed(wordsPerSecond: number): void {
        this.generationDelay = 1000 / wordsPerSecond;
    }

    setSimilarityThreshold(threshold: number): void {
        if (threshold >= 0 && threshold <= 1) {
            this.similarityThreshold = threshold;
        }
    }

    private getNextWord(currentWord: string): string | null {
        if (!this.transitions.has(currentWord)) return null;

        const possibleNextWords = this.transitions.get(currentWord)!;
        const words: string[] = [];
        const weights: number[] = [];

        for (const [word, count] of possibleNextWords.entries()) {
            words.push(word);
            weights.push(count);
        }

        return this.weightedRandom(words, weights);
    }

    private weightedRandom(items: string[], weights: number[]): string {
        const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
        let random = Math.random() * totalWeight;

        for (let i = 0; i < items.length; i++) {
            random -= weights[i];
            if (random <= 0) return items[i];
        }

        return items[0];
    }

    generate(maxLength: number = 50): string {
        if (this.startWords.length === 0) return "";

        const randomStartIndex = Math.floor(
            Math.random() * this.startWords.length
        );
        let currentWord = this.startWords[randomStartIndex];

        const result: string[] = [currentWord];

        for (let i = 1; i < maxLength; i++) {
            const nextWord = this.getNextWord(currentWord);
            if (nextWord === null) break;
            result.push(nextWord);
            currentWord = nextWord;
        }

        return result.join(" ");
    }
}

const qaDataset = [
    {
        question: "안녕하세요?",
        answer: "안녕하세요! WhatLm 입니다. 무엇을 도와드릴까요?",
    },
    {
        question: "너는 누구니?",
        answer: "저는 WhatLm이라는 질의응답 인공지능입니다. 질문에 대한 답변을 도와드릴게요.",
    },
    {
        question: "뭘 할 수 있어?",
        answer: "저는 질문에 답변하고, 간단한 대화를 나눌 수 있습니다. 무엇이든 물어보세요!",
    },
    {
        question: "오늘 날씨 어때?",
        answer: "제가 실시간 날씨 정보는 제공하지 못합니다. 날씨 앱이나 웹사이트를 확인해보시는 것이 좋을 것 같아요.",
    },
    {
        question: "고마워",
        answer: "천만에요! 더 필요한 것이 있으면 언제든지 말씀해주세요.",
    },
    {
        question: "도움이 필요해",
        answer: "네, 어떤 도움이 필요하신가요? 최대한 도와드리겠습니다.",
    },
    {
        question: "종료",
        answer: "대화를 종료합니다. 즐거운 하루 되세요!",
    },
];

async function interactiveMode() {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const readline = require("readline");
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    const model = new QnALanguageModel();

    const dataset = [
        "반갑습니다. WhatLm 입니다. 무엇을 도와드릴까요?",
        "안녕하세요. WhatLm 이에요. 도와드릴게 있을까요?",
        "좋은 하루에요. WhatLm 입니다. 원하는게 있나요?",
        "반가워요. 전 WhatLm 이라고 합니다. 무엇이든 물어보세요!",
        "만나서 영광이에요. 저는 WhatLm 이라고 합니다. 어떤 문제가 있나요?",
        "환영합니다. 제 이름은 WhatLm 이라고 합니다. 도와드릴게 있다면 언제든 물어보세요!",
        "반갑습니다. 저는 WhatLm 입니다. 무엇이든 도와드릴 수 있습니다.",
    ];

    for (const data of dataset) {
        model.trainMarkov(data);
    }

    for (const item of qaDataset) {
        model.trainQnA(item.question, item.answer);
    }

    const askQuestion = () => {
        rl.question("사용자: ", async (question: string) => {
            let answer = "";
            const generator = model.answerRealtime(question, 50);

            for await (const word of generator) {
                answer += (answer ? " " : "") + word;
                process.stdout.write("\r\x1b[K" + answer);
            }

            console.log("\n");

            if (answer) {
                model.trainQnA(question, answer);
            }

            askQuestion();
        });
    };

    askQuestion();
}

interactiveMode();
