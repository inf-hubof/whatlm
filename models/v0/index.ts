export class SimpleLanguageModel {
    private transitions: Map<string, Map<string, number>> = new Map();
    private startWords: string[] = [];
    private isGenerating: boolean = false;
    private generationDelay: number = 1000;

    train(text: string): void {
        const words = text.split(/\s+/).filter((word) => word.length > 0);

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

    async *generateRealtime(maxLength: number = 50) {
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

const dataset = [
    "반갑습니다. WhatLm 입니다. 무엇을 도와드릴까요?",
    "안녕하세요. WhatLm 이에요. 도와드릴게 있을까요?",
    "좋은 하루에요. WhatLm 입니다. 원하는게 있나요?",
    "반가워요. 전 WhatLm 이라고 합니다. 무엇이든 물어보세요!",
    "만나서 영광이에요. 저는 WhatLm 이라고 합니다. 어떤 문제가 있나요?",
    "환영합니다. 제 이름은 WhatLm 이라고 합니다. 도와드릴게 있다면 언제든 물어보세요!",
    "반갑습니다. 저는 WhatLm 입니다. 무엇이든 도와드릴 수 있습니다.",
];
async function demoRealtime() {
    const model = new SimpleLanguageModel();
    for (const data of dataset) model.train(data);

    const generator = model.generateRealtime(100);
    let text = "";

    for await (const word of generator) {
        text += (text ? " " : "") + word;

        process.stdout.write("\r\x1b[K" + text);

        // if (word === "무엇이든") {
        //     model.stopGeneration();
        //     break;
        // }
    }

    console.log();
}

demoRealtime();
