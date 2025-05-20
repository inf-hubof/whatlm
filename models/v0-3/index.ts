/**
 * AI의 도움 없이 직접 작성한 마르코프 체인 모델
 * 2025 05 20
 *
 * APACH 2.0 LICENSE
 */

export interface ModelProps {
    name: string;
}

export class Model {
    public name: string;

    public constructor(props: ModelProps) {
        this.name = props.name;
    }

    // 자연어 학습 메서드
    public trainNL(texts: string[]) {
        const words: string[] = [];

        for (const text of texts) {
            const extractedWords = text.split(/\s/);
            words.push(...extractedWords);
        }
    }
}

// 실행 함수
const main = async () => {
    const model = new Model({
        name: "whatlm_v0-3",
    });

    model.trainNL(["안녕하세요, 저는 고서온 이라고 합니다."]);
};

main();
