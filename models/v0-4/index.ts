/**
 * 외부 라이브러리 없이 구현한 트랜스포머 모델
 * 2025.05.20
 *
 * @author ice1github
 */

import fs from "fs/promises";

// 타입 정의 - 계산 효율성을 위한 단순 배열 사용 (TypedArray 사용 시 성능 향상 가능)
type Tensor1D = number[];
type Tensor2D = number[][];

// 행렬 연산 유틸리티 클래스 - WebGL/WASM 구현 시 성능 10-100배 향상 가능
class MatrixOps {
    // 행렬 곱셈 - O(n³) 연산, Strassen 알고리즘 사용 시 O(n^2.8) 가능
    static multiply(a: Tensor2D, b: Tensor2D): Tensor2D {
        const result: Tensor2D = [];
        const aRows = a.length;
        const aCols = a[0].length;
        const bCols = b[0].length;

        for (let i = 0; i < aRows; i++) {
            result[i] = [];
            for (let j = 0; j < bCols; j++) {
                let sum = 0;
                for (let k = 0; k < aCols; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    // 행렬 덧셈 - 단순 O(n²) 연산, 병렬화 가능
    static add(a: Tensor2D, b: Tensor2D): Tensor2D {
        const result: Tensor2D = [];
        const rows = a.length;
        const cols = a[0].length;

        for (let i = 0; i < rows; i++) {
            result[i] = [];
            for (let j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    // 스칼라 곱 - O(n²) 연산, SIMD 최적화 가능
    static scale(a: Tensor2D, scalar: number): Tensor2D {
        const result: Tensor2D = [];
        const rows = a.length;
        const cols = a[0].length;

        for (let i = 0; i < rows; i++) {
            result[i] = [];
            for (let j = 0; j < cols; j++) {
                result[i][j] = a[i][j] * scalar;
            }
        }
        return result;
    }

    // 소프트맥스 - 수치적 안정성을 위해 max값 차감 구현
    static softmax(x: Tensor1D): Tensor1D {
        const maxVal = Math.max(...x);
        const expValues = x.map((val) => Math.exp(val - maxVal));
        const sumExp = expValues.reduce((acc, val) => acc + val, 0);
        return expValues.map((val) => val / sumExp);
    }

    // 전치 행렬 - 캐시 친화적이지 않은 메모리 액세스 패턴
    static transpose(a: Tensor2D): Tensor2D {
        const rows = a.length;
        const cols = a[0].length;
        const result: Tensor2D = [];

        for (let j = 0; j < cols; j++) {
            result[j] = [];
            for (let i = 0; i < rows; i++) {
                result[j][i] = a[i][j];
            }
        }
        return result;
    }

    // 깊은 복사 - 참조 문제 방지
    static copy(a: Tensor2D): Tensor2D {
        return a.map((row) => [...row]);
    }

    // 랜덤 행렬 - He/Xavier 초기화 대신 단순 스케일링 사용
    static random(rows: number, cols: number, scale: number = 0.1): Tensor2D {
        const result: Tensor2D = [];
        for (let i = 0; i < rows; i++) {
            result[i] = [];
            for (let j = 0; j < cols; j++) {
                result[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
        return result;
    }

    // 행렬 평균 - 단일 패스 구현
    static mean(a: Tensor2D): number {
        let sum = 0;
        let count = 0;
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[i].length; j++) {
                sum += a[i][j];
                count++;
            }
        }
        return sum / count;
    }

    // 표준편차 계산 - 안정적인 구현, 베셀 보정 미적용
    static std(a: Tensor2D, mean: number): number {
        let sum = 0;
        let count = 0;
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[i].length; j++) {
                sum += Math.pow(a[i][j] - mean, 2);
                count++;
            }
        }
        return Math.sqrt(sum / count);
    }

    // 레이어 정규화 - 배치 정규화보다 트랜스포머에 적합
    static layerNorm(x: Tensor2D, epsilon: number = 1e-6): Tensor2D {
        const result: Tensor2D = [];

        for (let i = 0; i < x.length; i++) {
            // 행별 통계 계산 (배치가 아닌 피처 차원에서 정규화)
            const mean = x[i].reduce((sum, val) => sum + val, 0) / x[i].length;
            const variance =
                x[i].reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
                x[i].length;
            const std = Math.sqrt(variance + epsilon);

            // 정규화 적용
            result[i] = [];
            for (let j = 0; j < x[i].length; j++) {
                result[i][j] = (x[i][j] - mean) / std;
            }
        }

        return result;
    }

    // 행렬 차감 연산 - O(n²)
    static subtract(a: Tensor2D, b: Tensor2D): Tensor2D {
        const result: Tensor2D = [];
        const rows = a.length;
        const cols = a[0].length;

        for (let i = 0; i < rows; i++) {
            result[i] = [];
            for (let j = 0; j < cols; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    // 요소별 곱셈 (Hadamard) - 병렬화에 적합한 연산
    static elementWiseMultiply(a: Tensor2D, b: Tensor2D): Tensor2D {
        const result: Tensor2D = [];
        const rows = a.length;
        const cols = a[0].length;

        for (let i = 0; i < rows; i++) {
            result[i] = [];
            for (let j = 0; j < cols; j++) {
                result[i][j] = a[i][j] * b[i][j];
            }
        }
        return result;
    }

    // 마스킹 연산 - 어텐션에서 패딩/미래 토큰 차단용
    static mask(a: Tensor2D, mask: Tensor2D): Tensor2D {
        const result: Tensor2D = [];
        const rows = a.length;
        const cols = a[0].length;

        for (let i = 0; i < rows; i++) {
            result[i] = [];
            for (let j = 0; j < cols; j++) {
                result[i][j] = a[i][j] * mask[i][j];
            }
        }
        return result;
    }

    // 소프트맥스 미분 - 야코비안 행렬 계산 (-p_i*p_j for i!=j, p_i(1-p_i) for i=j)
    static softmaxDerivative(softmaxOutput: Tensor1D): Tensor2D {
        const size = softmaxOutput.length;
        const result: Tensor2D = [];

        for (let i = 0; i < size; i++) {
            result[i] = [];
            for (let j = 0; j < size; j++) {
                if (i === j) {
                    result[i][j] = softmaxOutput[i] * (1 - softmaxOutput[i]);
                } else {
                    result[i][j] = -softmaxOutput[i] * softmaxOutput[j];
                }
            }
        }
        return result;
    }

    // 행렬 총합 - 리듀스 연산
    static sum(a: Tensor2D): number {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[i].length; j++) {
                sum += a[i][j];
            }
        }
        return sum;
    }

    // 행 합계 - 차원 축소 연산
    static rowSum(a: Tensor2D): Tensor1D {
        const result: Tensor1D = [];
        for (let i = 0; i < a.length; i++) {
            let sum = 0;
            for (let j = 0; j < a[i].length; j++) {
                sum += a[i][j];
            }
            result.push(sum);
        }
        return result;
    }

    // 열 합계 - 차원 축소 연산 (전치 대신 직접 구현으로 효율성 증가)
    static colSum(a: Tensor2D): Tensor1D {
        const result: Tensor1D = new Array(a[0].length).fill(0);
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[i].length; j++) {
                result[j] += a[i][j];
            }
        }
        return result;
    }

    // 영행렬 생성 - 메모리 사전 할당
    static zeros(rows: number, cols: number): Tensor2D {
        const result: Tensor2D = [];
        for (let i = 0; i < rows; i++) {
            result[i] = new Array(cols).fill(0);
        }
        return result;
    }

    // 단위행렬 생성 - 대각선 요소만 1인 행렬
    static identity(size: number): Tensor2D {
        const result: Tensor2D = [];
        for (let i = 0; i < size; i++) {
            result[i] = new Array(size).fill(0);
            result[i][i] = 1;
        }
        return result;
    }
}

// 레이어 정규화 - Pre-LN 아키텍처 사용 (훈련 안정성 향상)
class LayerNorm {
    private gamma: Tensor1D; // 스케일 파라미터
    private beta: Tensor1D; // 시프트 파라미터
    private epsilon: number; // 0 나눗셈 방지용 작은 상수
    private dim: number; // 정규화 차원

    // 그래디언트 계산용 캐싱
    private lastInput: Tensor2D | null = null;
    private lastMean: Tensor1D | null = null;
    private lastVariance: Tensor1D | null = null;
    private lastNormalized: Tensor2D | null = null;
    private lastOutput: Tensor2D | null = null;

    constructor(dim: number, epsilon: number = 1e-6) {
        this.dim = dim;
        this.epsilon = epsilon;
        // 학습 파라미터 초기화 (표준 초기화)
        this.gamma = new Array(dim).fill(1); // 초기 스케일 1로 설정
        this.beta = new Array(dim).fill(0); // 초기 시프트 0으로 설정
    }

    // 순전파 - 배치 내 각 시퀀스에 대해 독립적으로 정규화
    public forward(input: Tensor2D): Tensor2D {
        this.lastInput = MatrixOps.copy(input);
        const output: Tensor2D = [];
        this.lastMean = [];
        this.lastVariance = [];
        this.lastNormalized = [];

        for (let i = 0; i < input.length; i++) {
            // 평균 계산 (E[x])
            const mean =
                input[i].reduce((sum, val) => sum + val, 0) / input[i].length;

            // 분산 계산 (Var[x])
            const variance =
                input[i].reduce(
                    (sum, val) => sum + Math.pow(val - mean, 2),
                    0
                ) / input[i].length;

            // 정규화 적용 ((x - E[x]) / sqrt(Var[x] + ε))
            const normalized: Tensor1D = [];
            for (let j = 0; j < input[i].length; j++) {
                normalized[j] =
                    (input[i][j] - mean) / Math.sqrt(variance + this.epsilon);
            }

            // 학습 가능한 파라미터 적용 (γ * normalized + β)
            output[i] = [];
            for (let j = 0; j < input[i].length; j++) {
                output[i][j] = this.gamma[j] * normalized[j] + this.beta[j];
            }

            // 역전파를 위한 중간값 저장
            this.lastMean.push(mean);
            this.lastVariance.push(variance);
            if (!this.lastNormalized[i]) this.lastNormalized[i] = [];
            this.lastNormalized[i] = [...normalized];
        }

        this.lastOutput = output;
        return output;
    }

    // 역전파 - 연쇄 법칙 적용하여 그래디언트 계산
    public backward(outputGradient: Tensor2D): {
        inputGradient: Tensor2D;
        gammaGradient: Tensor1D;
        betaGradient: Tensor1D;
    } {
        if (
            !this.lastInput ||
            !this.lastMean ||
            !this.lastVariance ||
            !this.lastNormalized ||
            !this.lastOutput
        ) {
            throw new Error("순전파가 먼저 실행되어야 합니다.");
        }

        const batchSize = outputGradient.length;
        const inputGradient: Tensor2D = [];
        const gammaGradient: Tensor1D = new Array(this.dim).fill(0);
        const betaGradient: Tensor1D = new Array(this.dim).fill(0);

        // γ, β에 대한 그래디언트 계산 - 배치 내 모든 요소에 대한 합
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.dim; j++) {
                gammaGradient[j] +=
                    outputGradient[i][j] * this.lastNormalized[i][j];
                betaGradient[j] += outputGradient[i][j];
            }
        }

        // 입력에 대한 그래디언트 계산 - 정규화 역전파
        for (let i = 0; i < batchSize; i++) {
            inputGradient[i] = [];
            const variance = this.lastVariance[i];
            const stdDev = Math.sqrt(variance + this.epsilon);

            // 중간 값 계산
            let sumDl_Dxhat = 0;
            let sumDl_Dxhat_xhat = 0;

            for (let j = 0; j < this.dim; j++) {
                const dl_dxhat = outputGradient[i][j] * this.gamma[j];
                sumDl_Dxhat += dl_dxhat;
                sumDl_Dxhat_xhat += dl_dxhat * this.lastNormalized[i][j];
            }

            for (let j = 0; j < this.dim; j++) {
                const xhat = this.lastNormalized[i][j];
                const dl_dxhat = outputGradient[i][j] * this.gamma[j];

                // 최종 그래디언트 계산 (dloss/dx = dloss/dxhat * [1/std - xhat*mean_correction - xhat*var_correction])
                inputGradient[i][j] =
                    (dl_dxhat -
                        sumDl_Dxhat / this.dim -
                        (xhat * sumDl_Dxhat_xhat) / this.dim) /
                    stdDev;
            }
        }

        return {
            inputGradient,
            gammaGradient,
            betaGradient,
        };
    }

    // 파라미터 접근자 - 모델 저장/로드용
    public getParams(): { gamma: Tensor1D; beta: Tensor1D } {
        return {
            gamma: [...this.gamma],
            beta: [...this.beta],
        };
    }

    // 파라미터 설정자 - 모델 저장/로드용
    public setParams(params: { gamma: Tensor1D; beta: Tensor1D }): void {
        this.gamma = [...params.gamma];
        this.beta = [...params.beta];
    }
}

// 활성화 함수 모음 - 비선형성 추가용
class Activation {
    // ReLU - max(0,x) 간단한 비선형 함수
    static relu(x: number): number {
        return Math.max(0, x);
    }

    // GELU - Gaussian Error Linear Unit, 트랜스포머에서 ReLU보다 성능 향상
    static gelu(x: number): number {
        return (
            0.5 *
            x *
            (1 +
                Math.tanh(
                    Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
                ))
        );
    }

    // GELU 행렬 적용 - 요소별 연산
    static applyGelu(x: Tensor2D): Tensor2D {
        const result: Tensor2D = [];
        for (let i = 0; i < x.length; i++) {
            result[i] = [];
            for (let j = 0; j < x[i].length; j++) {
                result[i][j] = this.gelu(x[i][j]);
            }
        }
        return result;
    }

    // GELU 미분 - 역전파용 (근사 계산으로 정확도와 속도 균형)
    static geluDerivative(x: number): number {
        // 가우시안 CDF 및 PDF 계산
        const cdf =
            0.5 *
            (1 +
                Math.tanh(
                    Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
                ));
        const pdf = Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI);

        // GELU 미분값 계산
        return (
            cdf +
            x *
                pdf *
                (1 -
                    Math.tanh(
                        Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
                    ) *
                        Math.tanh(
                            Math.sqrt(2 / Math.PI) *
                                (x + 0.044715 * Math.pow(x, 3))
                        )) *
                (Math.sqrt(2 / Math.PI) * (1 + 0.134145 * Math.pow(x, 2)))
        );
    }

    // GELU 미분 행렬 적용
    static applyGeluDerivative(x: Tensor2D): Tensor2D {
        const result: Tensor2D = [];
        for (let i = 0; i < x.length; i++) {
            result[i] = [];
            for (let j = 0; j < x[i].length; j++) {
                result[i][j] = this.geluDerivative(x[i][j]);
            }
        }
        return result;
    }
}

// 어텐션 헤드 인터페이스 - 구성 파라미터 정의
interface AttentionHeadProps {
    inputDim: number; // 입력 임베딩 차원
    headDim: number; // 어텐션 헤드 차원
}

// 어텐션 헤드 - 트랜스포머의 핵심 구성요소
class AttentionHead {
    private queryWeight: Tensor2D; // Q 변환 가중치
    private keyWeight: Tensor2D; // K 변환 가중치
    private valueWeight: Tensor2D; // V 변환 가중치
    private inputDim: number; // 입력 차원
    private headDim: number; // 헤드 차원

    // 역전파 계산용 중간값 캐싱
    private lastInput: Tensor2D | null = null;
    private lastQuery: Tensor2D | null = null;
    private lastKey: Tensor2D | null = null;
    private lastValue: Tensor2D | null = null;
    private lastScores: Tensor2D | null = null;
    private lastScaledScores: Tensor2D | null = null;
    private lastAttentionWeights: Tensor2D | null = null;
    private lastOutput: Tensor2D | null = null;

    constructor(props: AttentionHeadProps) {
        this.inputDim = props.inputDim;
        this.headDim = props.headDim;

        // 가중치 초기화 - 크기 스케일링으로 초기 분산 조정
        this.queryWeight = MatrixOps.random(this.inputDim, this.headDim, 0.02);
        this.keyWeight = MatrixOps.random(this.inputDim, this.headDim, 0.02);
        this.valueWeight = MatrixOps.random(this.inputDim, this.headDim, 0.02);
    }

    // 순전파 - 어텐션 계산 구현 (Q·K^T·V 플로우)
    public forward(input: Tensor2D): Tensor2D {
        // 입력 및 중간값 캐싱
        this.lastInput = MatrixOps.copy(input);

        // QKV 선형 투영
        this.lastQuery = MatrixOps.multiply(input, this.queryWeight);
        this.lastKey = MatrixOps.multiply(input, this.keyWeight);
        this.lastValue = MatrixOps.multiply(input, this.valueWeight);

        // 어텐션 스코어 계산 (Q·K^T) - O(n²d) 연산
        const keyTransposed = MatrixOps.transpose(this.lastKey);
        this.lastScores = MatrixOps.multiply(this.lastQuery, keyTransposed);

        // 스케일링 (√d_k로 나누기) - 그래디언트 안정화
        const scaleFactor = Math.sqrt(this.headDim);
        this.lastScaledScores = this.lastScores.map((row) =>
            row.map((val) => val / scaleFactor)
        );

        // 소프트맥스 적용 - 확률 분포 변환
        this.lastAttentionWeights = this.lastScaledScores.map((row) =>
            MatrixOps.softmax(row)
        );

        // 가중 합계 계산 (attention·V) - 컨텍스트 벡터 생성
        this.lastOutput = MatrixOps.multiply(
            this.lastAttentionWeights,
            this.lastValue
        );

        return this.lastOutput;
    }

    // 역전파 - 그래디언트 계산 (연쇄 법칙)
    public backward(outputGradient: Tensor2D): {
        inputGradient: Tensor2D;
        queryGradient: Tensor2D;
        keyGradient: Tensor2D;
        valueGradient: Tensor2D;
    } {
        if (
            !this.lastInput ||
            !this.lastQuery ||
            !this.lastKey ||
            !this.lastValue ||
            !this.lastScores ||
            !this.lastScaledScores ||
            !this.lastAttentionWeights ||
            !this.lastOutput
        ) {
            throw new Error("순전파가 먼저 실행되어야 합니다.");
        }

        // 1. V에 대한 그래디언트 계산 (∂L/∂V = A^T·∂L/∂output)
        const valueGradient = MatrixOps.transpose(
            MatrixOps.multiply(
                MatrixOps.transpose(outputGradient),
                this.lastAttentionWeights
            )
        );

        // 2. 어텐션 가중치에 대한 그래디언트 (∂L/∂A = ∂L/∂output·V^T)
        const attentionWeightsGradient = MatrixOps.multiply(
            outputGradient,
            MatrixOps.transpose(this.lastValue)
        );

        // 3. 스케일된 스코어에 대한 그래디언트 (소프트맥스 역전파, 야코비안 행렬 사용)
        const scaledScoresGradient: Tensor2D = [];
        for (let i = 0; i < attentionWeightsGradient.length; i++) {
            // 소프트맥스 자코비안 계산
            const softmaxJacobian = MatrixOps.softmaxDerivative(
                this.lastAttentionWeights[i]
            );
            // 그래디언트와 자코비안 곱
            const rowGradient = new Array(
                attentionWeightsGradient[i].length
            ).fill(0);
            for (let j = 0; j < attentionWeightsGradient[i].length; j++) {
                for (let k = 0; k < softmaxJacobian.length; k++) {
                    rowGradient[j] +=
                        attentionWeightsGradient[i][k] * softmaxJacobian[k][j];
                }
            }
            scaledScoresGradient.push(rowGradient);
        }

        // 4. 스케일링 역전파 (∂L/∂scores = ∂L/∂scaledScores / √d_k)
        const scaleFactor = Math.sqrt(this.headDim);
        const scoresGradient = scaledScoresGradient.map((row) =>
            row.map((val) => val / scaleFactor)
        );

        // 5. Q와 K에 대한 그래디언트
        const queryGradient = MatrixOps.multiply(scoresGradient, this.lastKey);
        const keyGradient = MatrixOps.multiply(
            MatrixOps.transpose(scoresGradient),
            this.lastQuery
        );

        // 6. 입력에 대한 그래디언트 (세 경로의 합)
        const inputGradientFromQuery = MatrixOps.multiply(
            queryGradient,
            MatrixOps.transpose(this.queryWeight)
        );
        const inputGradientFromKey = MatrixOps.multiply(
            keyGradient,
            MatrixOps.transpose(this.keyWeight)
        );
        const inputGradientFromValue = MatrixOps.multiply(
            valueGradient,
            MatrixOps.transpose(this.valueWeight)
        );

        // 세 경로 그래디언트 합산
        const inputGradient = MatrixOps.add(
            inputGradientFromQuery,
            MatrixOps.add(inputGradientFromKey, inputGradientFromValue)
        );

        // 가중치에 대한 그래디언트
        const queryWeightGradient = MatrixOps.multiply(
            MatrixOps.transpose(this.lastInput),
            queryGradient
        );
        const keyWeightGradient = MatrixOps.multiply(
            MatrixOps.transpose(this.lastInput),
            keyGradient
        );
        const valueWeightGradient = MatrixOps.multiply(
            MatrixOps.transpose(this.lastInput),
            valueGradient
        );

        return {
            inputGradient,
            queryGradient: queryWeightGradient,
            keyGradient: keyWeightGradient,
            valueGradient: valueWeightGradient,
        };
    }

    // 모델 파라미터 직렬화
    public getParams(): {
        queryWeight: Tensor2D;
        keyWeight: Tensor2D;
        valueWeight: Tensor2D;
    } {
        return {
            queryWeight: this.queryWeight,
            keyWeight: this.keyWeight,
            valueWeight: this.valueWeight,
        };
    }

    // 모델 파라미터 역직렬화
    public setParams(params: {
        queryWeight: Tensor2D;
        keyWeight: Tensor2D;
        valueWeight: Tensor2D;
    }): void {
        this.queryWeight = params.queryWeight;
        this.keyWeight = params.keyWeight;
        this.valueWeight = params.valueWeight;
    }
}

// 멀티헤드 어텐션 구성 인터페이스
interface MultiHeadAttentionProps {
    inputDim: number; // 입력 임베딩 차원
    numHeads: number; // 헤드 개수
    headDim: number; // 헤드당 차원
}

// 멀티헤드 어텐션 - 병렬 어텐션 헤드 구현
class MultiHeadAttention {
    private heads: AttentionHead[]; // 병렬 어텐션 헤드
    private outputWeight: Tensor2D; // 출력 투영 가중치
    private inputDim: number; // 입력 차원
    private numHeads: number; // 헤드 개수
    private headDim: number; // 헤드당 차원

    // 역전파용 캐싱
    private lastInput: Tensor2D | null = null;
    private lastHeadOutputs: Tensor2D[] | null = null;
    private lastConcatenated: Tensor2D | null = null;
    private lastOutput: Tensor2D | null = null;

    constructor(props: MultiHeadAttentionProps) {
        this.inputDim = props.inputDim;
        this.numHeads = props.numHeads;
        this.headDim = props.headDim;

        // 어텐션 헤드 초기화 - 병렬 처리 가능
        this.heads = [];
        for (let i = 0; i < this.numHeads; i++) {
            this.heads.push(
                new AttentionHead({
                    inputDim: this.inputDim,
                    headDim: this.headDim,
                })
            );
        }

        // 출력 프로젝션 레이어 초기화
        this.outputWeight = MatrixOps.random(
            this.numHeads * this.headDim,
            this.inputDim,
            0.02
        );
    }

    // 순전파 - 모든 헤드 병렬 계산 후 결과 연결 및 투영
    public forward(input: Tensor2D): Tensor2D {
        // 입력 캐싱
        this.lastInput = MatrixOps.copy(input);

        // 각 헤드 병렬 처리 (실제 구현은 직렬)
        this.lastHeadOutputs = this.heads.map((head) => head.forward(input));

        // 헤드 출력 연결 (concat) - [batch, seq_len, num_heads * head_dim]
        this.lastConcatenated = [];
        for (let i = 0; i < input.length; i++) {
            this.lastConcatenated[i] = [];
            for (let h = 0; h < this.numHeads; h++) {
                for (let j = 0; j < this.headDim; j++) {
                    this.lastConcatenated[i].push(
                        this.lastHeadOutputs[h][i][j]
                    );
                }
            }
        }

        // 최종 출력 투영 - 원래 모델 차원으로 복원
        this.lastOutput = MatrixOps.multiply(
            this.lastConcatenated,
            this.outputWeight
        );

        return this.lastOutput;
    }

    // 역전파 - 출력 투영 및 각 헤드 역전파
    public backward(outputGradient: Tensor2D): {
        inputGradient: Tensor2D;
        headsGradient: {
            queryGradient: Tensor2D;
            keyGradient: Tensor2D;
            valueGradient: Tensor2D;
        }[];
        outputWeightGradient: Tensor2D;
    } {
        if (
            !this.lastInput ||
            !this.lastHeadOutputs ||
            !this.lastConcatenated ||
            !this.lastOutput
        ) {
            throw new Error("순전파가 먼저 실행되어야 합니다.");
        }

        // 1. 출력 가중치에 대한 그래디언트
        const outputWeightGradient = MatrixOps.multiply(
            MatrixOps.transpose(this.lastConcatenated),
            outputGradient
        );

        // 2. 연결된 헤드 출력에 대한 그래디언트
        const concatenatedGradient = MatrixOps.multiply(
            outputGradient,
            MatrixOps.transpose(this.outputWeight)
        );

        // 3. 개별 헤드 출력에 대한 그래디언트 분배
        const headOutputsGradient: Tensor2D[] = [];
        for (let h = 0; h < this.numHeads; h++) {
            headOutputsGradient[h] = [];
            for (let i = 0; i < this.lastInput.length; i++) {
                headOutputsGradient[h][i] = [];
                for (let j = 0; j < this.headDim; j++) {
                    const index = h * this.headDim + j;
                    headOutputsGradient[h][i][j] =
                        concatenatedGradient[i][index];
                }
            }
        }

        // 4. 각 헤드 역전파 - 병렬 처리 가능
        const headsBackwardResults = headOutputsGradient.map(
            (gradient, index) => this.heads[index].backward(gradient)
        );

        // 5. 입력 그래디언트 합산 (모든 헤드로부터)
        let inputGradient = headsBackwardResults[0].inputGradient;
        for (let i = 1; i < headsBackwardResults.length; i++) {
            inputGradient = MatrixOps.add(
                inputGradient,
                headsBackwardResults[i].inputGradient
            );
        }

        // 6. 헤드 가중치 그래디언트 수집
        const headsGradient = headsBackwardResults.map((result) => ({
            queryGradient: result.queryGradient,
            keyGradient: result.keyGradient,
            valueGradient: result.valueGradient,
        }));

        return {
            inputGradient,
            headsGradient,
            outputWeightGradient,
        };
    }

    // 모델 파라미터 접근자
    public getParams(): {
        heads: {
            queryWeight: Tensor2D;
            keyWeight: Tensor2D;
            valueWeight: Tensor2D;
        }[];
        outputWeight: Tensor2D;
    } {
        return {
            heads: this.heads.map((head) => head.getParams()),
            outputWeight: this.outputWeight,
        };
    }

    // 모델 파라미터 설정자
    public setParams(params: {
        heads: {
            queryWeight: Tensor2D;
            keyWeight: Tensor2D;
            valueWeight: Tensor2D;
        }[];
        outputWeight: Tensor2D;
    }): void {
        for (let i = 0; i < this.heads.length; i++) {
            this.heads[i].setParams(params.heads[i]);
        }
        this.outputWeight = params.outputWeight;
    }
}

// 피드포워드 네트워크 인터페이스
interface FeedForwardProps {
    inputDim: number; // 입력 차원
    hiddenDim: number; // 히든 차원 (보통 입력의 4배)
}

// 피드포워드 네트워크 - MLP 구현 (GELU 활성화)
class FeedForward {
    private weight1: Tensor2D; // 첫 번째 레이어 가중치
    private weight2: Tensor2D; // 두 번째 레이어 가중치
    private inputDim: number; // 입력 차원
    private hiddenDim: number; // 히든 차원

    // 역전파용 캐싱
    private lastInput: Tensor2D | null = null;
    private lastHidden: Tensor2D | null = null;
    private lastActivated: Tensor2D | null = null;
    private lastOutput: Tensor2D | null = null;

    constructor(props: FeedForwardProps) {
        this.inputDim = props.inputDim;
        this.hiddenDim = props.hiddenDim;

        // 가중치 초기화 - XavierNormal/HeNormal 대신 단순 스케일링
        this.weight1 = MatrixOps.random(this.inputDim, this.hiddenDim, 0.02);
        this.weight2 = MatrixOps.random(this.hiddenDim, this.inputDim, 0.02);
    }

    // 순전파 - 확장 후 압축하는 구조 (병목 아키텍처)
    public forward(input: Tensor2D): Tensor2D {
        // 입력 캐싱
        this.lastInput = MatrixOps.copy(input);

        // 확장 레이어 (inputDim → hiddenDim)
        this.lastHidden = MatrixOps.multiply(input, this.weight1);
        this.lastActivated = Activation.applyGelu(this.lastHidden);

        // 압축 레이어 (hiddenDim → inputDim)
        this.lastOutput = MatrixOps.multiply(this.lastActivated, this.weight2);

        return this.lastOutput;
    }

    // 역전파 - 층별 그래디언트 계산
    public backward(outputGradient: Tensor2D): {
        inputGradient: Tensor2D;
        weight1Gradient: Tensor2D;
        weight2Gradient: Tensor2D;
    } {
        if (
            !this.lastInput ||
            !this.lastHidden ||
            !this.lastActivated ||
            !this.lastOutput
        ) {
            throw new Error("순전파가 먼저 실행되어야 합니다.");
        }

        // 1. weight2에 대한 그래디언트 (∂L/∂W2 = activated^T · ∂L/∂output)
        const weight2Gradient = MatrixOps.multiply(
            MatrixOps.transpose(this.lastActivated),
            outputGradient
        );

        // 2. 활성화 출력에 대한 그래디언트 (∂L/∂activated = ∂L/∂output · W2^T)
        const activatedGradient = MatrixOps.multiply(
            outputGradient,
            MatrixOps.transpose(this.weight2)
        );

        // 3. GELU 비선형성 역전파
        const geluDerivative = Activation.applyGeluDerivative(this.lastHidden);
        const hiddenGradient = MatrixOps.elementWiseMultiply(
            activatedGradient,
            geluDerivative
        );

        // 4. weight1에 대한 그래디언트 (∂L/∂W1 = input^T · ∂L/∂hidden)
        const weight1Gradient = MatrixOps.multiply(
            MatrixOps.transpose(this.lastInput),
            hiddenGradient
        );

        // 5. 입력에 대한 그래디언트 (∂L/∂input = ∂L/∂hidden · W1^T)
        const inputGradient = MatrixOps.multiply(
            hiddenGradient,
            MatrixOps.transpose(this.weight1)
        );

        return {
            inputGradient,
            weight1Gradient,
            weight2Gradient,
        };
    }

    // 모델 파라미터 접근자
    public getParams(): { weight1: Tensor2D; weight2: Tensor2D } {
        return {
            weight1: this.weight1,
            weight2: this.weight2,
        };
    }

    // 모델 파라미터 설정자
    public setParams(params: { weight1: Tensor2D; weight2: Tensor2D }): void {
        this.weight1 = params.weight1;
        this.weight2 = params.weight2;
    }
}

// 포지셔널 인코딩 인터페이스
interface PositionalEncodingProps {
    maxSeqLength: number; // 최대 시퀀스 길이
    embeddingDim: number; // 임베딩 차원
}

// 포지셔널 인코딩 - 위치 정보 주입 (사인/코사인 함수 기반)
class PositionalEncoding {
    private encodings: Tensor2D; // 사전 계산된 인코딩 값
    private maxSeqLength: number; // 최대 시퀀스 길이
    private embeddingDim: number; // 임베딩 차원 크기

    // 역전파용 캐싱
    private lastInput: Tensor2D | null = null;
    private lastOutput: Tensor2D | null = null;

    constructor(props: PositionalEncodingProps) {
        this.maxSeqLength = props.maxSeqLength;
        this.embeddingDim = props.embeddingDim;

        // 사전 계산된 위치 인코딩 생성
        this.encodings = this.generateEncodings();
    }

    // 포지셔널 인코딩 생성 - 상대적 위치 정보 인코딩
    private generateEncodings(): Tensor2D {
        const encodings: Tensor2D = [];

        for (let pos = 0; pos < this.maxSeqLength; pos++) {
            const posEncoding: Tensor1D = [];

            for (let i = 0; i < this.embeddingDim; i++) {
                if (i % 2 === 0) {
                    // 짝수 차원: sin 함수
                    posEncoding.push(
                        Math.sin(pos / Math.pow(10000, i / this.embeddingDim))
                    );
                } else {
                    // 홀수 차원: cos 함수 (같은 주파수의 직교 신호)
                    posEncoding.push(
                        Math.cos(
                            pos / Math.pow(10000, (i - 1) / this.embeddingDim)
                        )
                    );
                }
            }

            encodings.push(posEncoding);
        }

        return encodings;
    }

    // 위치 인코딩 적용 - 임베딩에 위치 정보 더하기
    public addPositionalEncoding(embeddings: Tensor2D): Tensor2D {
        this.lastInput = MatrixOps.copy(embeddings);
        const result: Tensor2D = [];
        const seqLength = Math.min(embeddings.length, this.maxSeqLength);

        for (let i = 0; i < seqLength; i++) {
            result[i] = [];
            for (let j = 0; j < this.embeddingDim; j++) {
                result[i][j] = embeddings[i][j] + this.encodings[i][j];
            }
        }

        this.lastOutput = result;
        return result;
    }

    // 역전파 - 위치 인코딩은 학습 파라미터 없음 (그래디언트 단순 전달)
    public backward(outputGradient: Tensor2D): { inputGradient: Tensor2D } {
        if (!this.lastInput || !this.lastOutput) {
            throw new Error("순전파가 먼저 실행되어야 합니다.");
        }

        // 그래디언트 그대로 전달 (덧셈 연산의 역전파)
        return { inputGradient: outputGradient };
    }

    // 모델 파라미터 접근자
    public getParams(): { encodings: Tensor2D } {
        return {
            encodings: this.encodings,
        };
    }

    // 모델 파라미터 설정자
    public setParams(params: { encodings: Tensor2D }): void {
        this.encodings = params.encodings;
    }
}

// 트랜스포머 레이어 인터페이스
interface TransformerLayerProps {
    embeddingDim: number; // 임베딩 차원
    numHeads: number; // 어텐션 헤드 수
    feedForwardDim: number; // 피드포워드 히든 차원
}

// 트랜스포머 레이어 - 어텐션과 피드포워드 네트워크 결합
class TransformerLayer {
    private attention: MultiHeadAttention; // 멀티헤드 어텐션 모듈
    private feedForward: FeedForward; // 피드포워드 네트워크
    private layerNorm1: LayerNorm; // 첫 번째 레이어 정규화
    private layerNorm2: LayerNorm; // 두 번째 레이어 정규화
    private embeddingDim: number; // 임베딩 차원

    // 역전파용 캐싱
    private lastInput: Tensor2D | null = null;
    private lastAttentionOutput: Tensor2D | null = null;
    private lastResidual1: Tensor2D | null = null;
    private lastNormalized1: Tensor2D | null = null;
    private lastFeedForwardOutput: Tensor2D | null = null;
    private lastResidual2: Tensor2D | null = null;
    private lastOutput: Tensor2D | null = null;

    constructor(props: TransformerLayerProps) {
        this.embeddingDim = props.embeddingDim;

        // 컴포넌트 초기화
        const headDim = Math.floor(this.embeddingDim / props.numHeads);
        this.attention = new MultiHeadAttention({
            inputDim: this.embeddingDim,
            numHeads: props.numHeads,
            headDim: headDim,
        });

        this.feedForward = new FeedForward({
            inputDim: this.embeddingDim,
            hiddenDim: props.feedForwardDim,
        });

        // 레이어 정규화 초기화 (Pre-LN 아키텍처)
        this.layerNorm1 = new LayerNorm(this.embeddingDim);
        this.layerNorm2 = new LayerNorm(this.embeddingDim);
    }

    // 순전파 - 전체 트랜스포머 레이어 처리
    public forward(input: Tensor2D): Tensor2D {
        // 입력 캐싱
        this.lastInput = MatrixOps.copy(input);

        // 1. 셀프 어텐션 서브레이어
        this.lastAttentionOutput = this.attention.forward(input);

        // 2. 잔차 연결 (스킵 커넥션)
        this.lastResidual1 = MatrixOps.add(input, this.lastAttentionOutput);

        // 3. 레이어 정규화
        this.lastNormalized1 = this.layerNorm1.forward(this.lastResidual1);

        // 4. 피드포워드 서브레이어
        this.lastFeedForwardOutput = this.feedForward.forward(
            this.lastNormalized1
        );

        // 5. 잔차 연결 (스킵 커넥션)
        this.lastResidual2 = MatrixOps.add(
            this.lastNormalized1,
            this.lastFeedForwardOutput
        );

        // 6. 레이어 정규화
        this.lastOutput = this.layerNorm2.forward(this.lastResidual2);

        return this.lastOutput;
    }

    // 역전파 - 복합 그래디언트 계산
    public backward(outputGradient: Tensor2D): {
        inputGradient: Tensor2D;
        attentionGradient: ReturnType<MultiHeadAttention["backward"]>;
        feedForwardGradient: ReturnType<FeedForward["backward"]>;
        layerNorm1Gradient: ReturnType<LayerNorm["backward"]>;
        layerNorm2Gradient: ReturnType<LayerNorm["backward"]>;
    } {
        if (
            !this.lastInput ||
            !this.lastAttentionOutput ||
            !this.lastResidual1 ||
            !this.lastNormalized1 ||
            !this.lastFeedForwardOutput ||
            !this.lastResidual2 ||
            !this.lastOutput
        ) {
            throw new Error("순전파가 먼저 실행되어야 합니다.");
        }

        // 1. layerNorm2 역전파
        const layerNorm2Gradient = this.layerNorm2.backward(outputGradient);

        // 2. 두 번째 잔차 연결 역전파 (그래디언트 분기)
        const feedForwardOutputGradient = layerNorm2Gradient.inputGradient;
        const normalized1Gradient = MatrixOps.copy(feedForwardOutputGradient);

        // 3. feedForward 역전파
        const feedForwardGradient = this.feedForward.backward(
            feedForwardOutputGradient
        );

        // 4. 첫 번째 잔차 연결 그래디언트 합산
        for (let i = 0; i < normalized1Gradient.length; i++) {
            for (let j = 0; j < normalized1Gradient[i].length; j++) {
                normalized1Gradient[i][j] +=
                    feedForwardGradient.inputGradient[i][j];
            }
        }

        // 5. layerNorm1 역전파
        const layerNorm1Gradient =
            this.layerNorm1.backward(normalized1Gradient);

        // 6. 첫 번째 잔차 연결 역전파 (그래디언트 분기)
        const attentionOutputGradient = layerNorm1Gradient.inputGradient;
        const inputGradient = MatrixOps.copy(attentionOutputGradient);

        // 7. attention 역전파
        const attentionGradient = this.attention.backward(
            attentionOutputGradient
        );

        // 8. 입력 그래디언트 합산
        for (let i = 0; i < inputGradient.length; i++) {
            for (let j = 0; j < inputGradient[i].length; j++) {
                inputGradient[i][j] += attentionGradient.inputGradient[i][j];
            }
        }

        return {
            inputGradient,
            attentionGradient,
            feedForwardGradient,
            layerNorm1Gradient,
            layerNorm2Gradient,
        };
    }

    // 모델 파라미터 접근자
    public getParams(): {
        attention: ReturnType<MultiHeadAttention["getParams"]>;
        feedForward: ReturnType<FeedForward["getParams"]>;
        layerNorm1: ReturnType<LayerNorm["getParams"]>;
        layerNorm2: ReturnType<LayerNorm["getParams"]>;
    } {
        return {
            attention: this.attention.getParams(),
            feedForward: this.feedForward.getParams(),
            layerNorm1: this.layerNorm1.getParams(),
            layerNorm2: this.layerNorm2.getParams(),
        };
    }

    // 모델 파라미터 설정자
    public setParams(params: {
        attention: ReturnType<MultiHeadAttention["getParams"]>;
        feedForward: ReturnType<FeedForward["getParams"]>;
        layerNorm1: ReturnType<LayerNorm["getParams"]>;
        layerNorm2: ReturnType<LayerNorm["getParams"]>;
    }): void {
        this.attention.setParams(params.attention);
        this.feedForward.setParams(params.feedForward);
        this.layerNorm1.setParams(params.layerNorm1);
        this.layerNorm2.setParams(params.layerNorm2);
    }
}

// 임베딩 레이어 인터페이스
interface EmbeddingProps {
    vocabSize: number; // 어휘 크기
    embeddingDim: number; // 임베딩 차원
}

// 임베딩 클래스 - 어휘 매핑 및 토큰 벡터화
class Embedding {
    private weight: Tensor2D; // 임베딩 가중치 행렬
    private vocabSize: number; // 어휘 크기
    private embeddingDim: number; // 임베딩 차원
    private tokenToIndex: Map<string, number>; // 토큰→인덱스 매핑
    private indexToToken: Map<number, string>; // 인덱스→토큰 매핑

    // 역전파용 캐싱
    private lastIndices: number[] | null = null;
    private lastOutput: Tensor2D | null = null;

    constructor(props: EmbeddingProps) {
        this.vocabSize = props.vocabSize;
        this.embeddingDim = props.embeddingDim;

        // 임베딩 가중치 초기화 (정규분포 기반)
        this.weight = MatrixOps.random(this.vocabSize, this.embeddingDim, 0.02);

        // 토큰-인덱스 매핑 초기화
        this.tokenToIndex = new Map<string, number>();
        this.indexToToken = new Map<number, string>();
    }

    // 어휘 구축 - 텍스트 말뭉치로부터 토큰 추출
    public buildVocab(texts: string[]): void {
        const tokens = new Set<string>();

        // 텍스트에서 공백 기준 단어 토큰화 (실제로는 BPE/WordPiece 권장)
        for (const text of texts) {
            const words = text.split(/\s+/);
            for (const word of words) {
                tokens.add(word);
            }
        }

        // 빈도수 기반 토큰 선택 (실제로는 빈도 정렬 필요)
        let index = 0;
        for (const token of tokens) {
            if (index < this.vocabSize - 2) {
                // 특수 토큰 공간 예약
                this.tokenToIndex.set(token, index);
                this.indexToToken.set(index, token);
                index++;
            }
        }

        // 특수 토큰 추가 (PAD, UNK) - 실제로는 BOS/EOS도 필요
        this.tokenToIndex.set("<PAD>", this.vocabSize - 2);
        this.indexToToken.set(this.vocabSize - 2, "<PAD>");
        this.tokenToIndex.set("<UNK>", this.vocabSize - 1);
        this.indexToToken.set(this.vocabSize - 1, "<UNK>");
    }

    // 텍스트→인덱스 변환 (토큰화)
    public textToIndices(text: string): number[] {
        const words = text.split(/\s+/);
        return words.map((word) => {
            if (this.tokenToIndex.has(word)) {
                return this.tokenToIndex.get(word)!;
            } else {
                return this.tokenToIndex.get("<UNK>")!; // 미등록 토큰
            }
        });
    }

    // 인덱스→텍스트 변환 (디코딩)
    public indicesToText(indices: number[]): string {
        return indices
            .map((index) => {
                if (this.indexToToken.has(index)) {
                    return this.indexToToken.get(index)!;
                } else {
                    return this.indexToToken.get(this.vocabSize - 1)!; // UNK
                }
            })
            .join(" ");
    }

    // 순전파 - 인덱스→벡터 변환 (룩업 테이블)
    public forward(indices: number[]): Tensor2D {
        this.lastIndices = [...indices];
        const embeddings: Tensor2D = [];

        // 각 인덱스에 대한 임베딩 벡터 룩업
        for (const index of indices) {
            embeddings.push([...this.weight[index]]);
        }

        this.lastOutput = embeddings;
        return embeddings;
    }

    // 역전파 - 임베딩 가중치 업데이트
    public backward(outputGradient: Tensor2D): { weightGradient: Tensor2D } {
        if (!this.lastIndices || !this.lastOutput) {
            throw new Error("순전파가 먼저 실행되어야 합니다.");
        }

        // 가중치 그래디언트 초기화 (희소 매트릭스 최적화 가능)
        const weightGradient = MatrixOps.zeros(
            this.vocabSize,
            this.embeddingDim
        );

        // 사용된 토큰에 대해서만 그래디언트 누적
        for (let i = 0; i < this.lastIndices.length; i++) {
            const index = this.lastIndices[i];
            for (let j = 0; j < this.embeddingDim; j++) {
                weightGradient[index][j] += outputGradient[i][j];
            }
        }

        return { weightGradient };
    }

    // 모델 파라미터 접근자
    public getParams(): {
        weight: Tensor2D;
        tokenToIndex: [string, number][];
        indexToToken: [number, string][];
    } {
        return {
            weight: this.weight,
            tokenToIndex: Array.from(this.tokenToIndex.entries()),
            indexToToken: Array.from(this.indexToToken.entries()),
        };
    }

    // 모델 파라미터 설정자
    public setParams(params: {
        weight: Tensor2D;
        tokenToIndex: [string, number][];
        indexToToken: [number, string][];
    }): void {
        this.weight = params.weight;
        this.tokenToIndex = new Map(params.tokenToIndex);
        this.indexToToken = new Map(params.indexToToken);
    }
}

// Adam 옵티마이저 - 적응적 모멘텀 기반 최적화
class AdamOptimizer {
    private alpha: number; // 학습률
    private beta1: number; // 1차 모멘텀 계수
    private beta2: number; // 2차 모멘텀 계수
    private epsilon: number; // 수치 안정성 상수
    private t: number; // 타임스텝
    private m: Map<string, Tensor2D>; // 1차 모멘텀 (속도)
    private v: Map<string, Tensor2D>; // 2차 모멘텀 (가속도)

    constructor(
        alpha: number = 0.001,
        beta1: number = 0.9,
        beta2: number = 0.999,
        epsilon: number = 1e-8
    ) {
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
        this.m = new Map<string, Tensor2D>();
        this.v = new Map<string, Tensor2D>();
    }

    // 파라미터 업데이트 - Adam 알고리즘 구현
    public update(
        key: string,
        params: Tensor2D,
        gradients: Tensor2D
    ): Tensor2D {
        this.t += 1;

        // 모멘텀, 속도 초기화 (첫 번째 호출 시)
        if (!this.m.has(key)) {
            this.m.set(key, MatrixOps.scale(params, 0));
            this.v.set(key, MatrixOps.scale(params, 0));
        }

        // 모멘텀 및 속도 추출
        const m = this.m.get(key)!;
        const v = this.v.get(key)!;

        // 1차 모멘텀 업데이트 (이동 평균) - m_t = β₁·m_{t-1} + (1-β₁)·g_t
        const m_t: Tensor2D = [];
        // 2차 모멘텀 업데이트 (이동 분산) - v_t = β₂·v_{t-1} + (1-β₂)·g_t²
        const v_t: Tensor2D = [];

        for (let i = 0; i < params.length; i++) {
            m_t[i] = [];
            v_t[i] = [];
            for (let j = 0; j < params[0].length; j++) {
                m_t[i][j] =
                    this.beta1 * m[i][j] + (1 - this.beta1) * gradients[i][j];
                v_t[i][j] =
                    this.beta2 * v[i][j] +
                    (1 - this.beta2) * (gradients[i][j] * gradients[i][j]);
            }
        }

        this.m.set(key, m_t);
        this.v.set(key, v_t);

        // 바이어스 보정 - 초기 타임스텝에서의 추정치 불균형 해소
        const m_hat: Tensor2D = [];
        const v_hat: Tensor2D = [];

        for (let i = 0; i < params.length; i++) {
            m_hat[i] = [];
            v_hat[i] = [];
            for (let j = 0; j < params[0].length; j++) {
                // m_hat = m_t / (1-β₁ᵗ)
                m_hat[i][j] = m_t[i][j] / (1 - Math.pow(this.beta1, this.t));
                // v_hat = v_t / (1-β₂ᵗ)
                v_hat[i][j] = v_t[i][j] / (1 - Math.pow(this.beta2, this.t));
            }
        }

        // 파라미터 업데이트 - θ_t = θ_{t-1} - α·m_hat/√(v_hat)+ε
        const updated: Tensor2D = [];
        for (let i = 0; i < params.length; i++) {
            updated[i] = [];
            for (let j = 0; j < params[0].length; j++) {
                updated[i][j] =
                    params[i][j] -
                    (this.alpha * m_hat[i][j]) /
                        (Math.sqrt(v_hat[i][j]) + this.epsilon);
            }
        }

        return updated;
    }
}

// 손실 함수 및 그래디언트 계산
class Loss {
    // 크로스 엔트로피 손실 - 분류 태스크용 (-Σy*log(p))
    static crossEntropy(logits: Tensor2D, targets: number[]): number {
        let loss = 0;
        for (let i = 0; i < logits.length; i++) {
            // 수치적 안정성을 위한 클리핑
            loss -= Math.log(Math.max(logits[i][targets[i]], 1e-10));
        }
        return loss / logits.length; // 배치 평균
    }

    // 크로스 엔트로피 그래디언트 - 소프트맥스 미분과 결합
    static crossEntropyGradient(logits: Tensor2D, targets: number[]): Tensor2D {
        const gradients: Tensor2D = [];
        for (let i = 0; i < logits.length; i++) {
            gradients[i] = [...logits[i]]; // 소프트맥스 출력 복사
            gradients[i][targets[i]] -= 1; // 타깃 위치에서 1 빼기
        }
        return gradients;
    }
}

// 트랜스포머 모델 인터페이스
export interface TransformerProps {
    name: string; // 모델 식별자
    vocabSize: number; // 어휘 크기
    embeddingDim: number; // 임베딩 차원
    numLayers: number; // 트랜스포머 레이어 수
    numHeads: number; // 어텐션 헤드 수
    feedForwardDim: number; // 피드포워드 히든 차원
    maxSeqLength: number; // 최대 시퀀스 길이
}

// 트랜스포머 모델 - 메인 클래스
export class Transformer {
    public name: string; // 모델 식별자
    private embedding: Embedding; // 토큰 임베딩
    private positionalEncoding: PositionalEncoding; // 위치 인코딩
    private layers: TransformerLayer[]; // 트랜스포머 레이어 스택
    private outputLayer: Tensor2D; // 출력 투영 레이어
    private vocabSize: number; // 어휘 크기
    private embeddingDim: number; // 임베딩 차원
    private numLayers: number; // 레이어 수
    private numHeads: number; // 어텐션 헤드 수
    private feedForwardDim: number; // 피드포워드 차원
    private maxSeqLength: number; // 최대 시퀀스 길이

    // 옵티마이저
    private optimizer: AdamOptimizer; // 파라미터 업데이트용

    // 역전파용 캐싱
    private lastIndices: number[] | null = null;
    private lastEmbedding: Tensor2D | null = null;
    private lastPositionalEncoding: Tensor2D | null = null;
    private lastLayerOutputs: Tensor2D[] | null = null;
    private lastOutput: Tensor2D | null = null;
    private lastLogits: Tensor2D | null = null;
    private lastProbabilities: Tensor2D | null = null;

    constructor(props: TransformerProps) {
        this.name = props.name;
        this.vocabSize = props.vocabSize;
        this.embeddingDim = props.embeddingDim;
        this.numLayers = props.numLayers;
        this.numHeads = props.numHeads;
        this.feedForwardDim = props.feedForwardDim;
        this.maxSeqLength = props.maxSeqLength;

        // 임베딩 초기화
        this.embedding = new Embedding({
            vocabSize: this.vocabSize,
            embeddingDim: this.embeddingDim,
        });

        // 위치 인코딩 초기화
        this.positionalEncoding = new PositionalEncoding({
            maxSeqLength: this.maxSeqLength,
            embeddingDim: this.embeddingDim,
        });

        // 트랜스포머 레이어 초기화
        this.layers = [];
        for (let i = 0; i < this.numLayers; i++) {
            this.layers.push(
                new TransformerLayer({
                    embeddingDim: this.embeddingDim,
                    numHeads: this.numHeads,
                    feedForwardDim: this.feedForwardDim,
                })
            );
        }

        // 출력 프로젝션 초기화 (임베딩 차원 → 어휘 크기)
        this.outputLayer = MatrixOps.random(
            this.embeddingDim,
            this.vocabSize,
            0.02
        );

        // 옵티마이저 초기화
        this.optimizer = new AdamOptimizer(0.001);
    }

    // 언어 모델 학습 - NL 태스크 전용
    public trainNL(
        texts: string[], // 학습 텍스트 말뭉치
        epochs: number = 10, // 반복 횟수
        batchSize: number = 4, // 배치 크기
        learningRate: number = 0.001 // 학습률
    ): void {
        console.log("transformer start learning...");
        console.log(
            `data: ${texts.length}, epoch: ${epochs}, batchsize: ${batchSize}, percentage: ${learningRate}`
        );

        // 1. 어휘 구축 - 토큰화 및 인덱싱
        this.embedding.buildVocab(texts);
        console.log(`vocabsize: ${this.vocabSize}`);

        // 2. 학습 데이터 준비 - 입력/타겟 쌍 생성
        const trainingData: { input: number[]; target: number[] }[] = [];

        for (const text of texts) {
            // 텍스트 토큰화
            const tokens = this.embedding.textToIndices(text);

            // 컨텍스트 윈도우 슬라이딩 (최소 2 토큰 이상)
            if (tokens.length >= 2) {
                for (let i = 0; i < tokens.length - 1; i++) {
                    // 가변 컨텍스트 길이 적용 (최대 길이 제한)
                    const maxContext = Math.min(i + 1, this.maxSeqLength);
                    const input = tokens.slice(i + 1 - maxContext, i + 1);
                    const target = tokens[i + 1];
                    trainingData.push({
                        input,
                        target: [target], // 다음 토큰 예측
                    });
                }
            }
        }

        // 데이터 검증
        if (trainingData.length === 0) {
            console.error(
                "훈련 데이터가 충분하지 않습니다. 더 긴 텍스트가 필요합니다."
            );
            return;
        }

        console.log(`learning-sample: ${trainingData.length}`);

        // 3. 에폭 반복 - 전체 데이터셋 학습
        for (let epoch = 0; epoch < epochs; epoch++) {
            const startTime = Date.now();
            let totalLoss = 0;
            let batchCount = 0;

            // 데이터 셔플링 - 과적합 방지
            this.shuffle(trainingData);

            // 배치 단위 처리
            for (
                let batchStart = 0;
                batchStart < trainingData.length;
                batchStart += batchSize
            ) {
                const batchEnd = Math.min(
                    batchStart + batchSize,
                    trainingData.length
                );
                const batch = trainingData.slice(batchStart, batchEnd);

                // 배치 손실 초기화
                let batchLoss = 0;

                // 그래디언트 누적 초기화 (미니배치 SGD)
                const embeddingGradients = MatrixOps.zeros(
                    this.vocabSize,
                    this.embeddingDim
                );
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const layerGradients: any = Array(this.layers.length)
                    .fill(null)
                    .map(() => ({
                        attentionGradient: {
                            headsGradient: Array(this.numHeads)
                                .fill(null)
                                .map(() => ({
                                    queryGradient: MatrixOps.zeros(
                                        this.embeddingDim,
                                        this.embeddingDim / this.numHeads
                                    ),
                                    keyGradient: MatrixOps.zeros(
                                        this.embeddingDim,
                                        this.embeddingDim / this.numHeads
                                    ),
                                    valueGradient: MatrixOps.zeros(
                                        this.embeddingDim,
                                        this.embeddingDim / this.numHeads
                                    ),
                                })),
                            outputWeightGradient: MatrixOps.zeros(
                                this.numHeads *
                                    (this.embeddingDim / this.numHeads),
                                this.embeddingDim
                            ),
                        },
                        feedForwardGradient: {
                            weight1Gradient: MatrixOps.zeros(
                                this.embeddingDim,
                                this.feedForwardDim
                            ),
                            weight2Gradient: MatrixOps.zeros(
                                this.feedForwardDim,
                                this.embeddingDim
                            ),
                        },
                        layerNorm1Gradient: {
                            gammaGradient: new Array(this.embeddingDim).fill(0),
                            betaGradient: new Array(this.embeddingDim).fill(0),
                        },
                        layerNorm2Gradient: {
                            gammaGradient: new Array(this.embeddingDim).fill(0),
                            betaGradient: new Array(this.embeddingDim).fill(0),
                        },
                        inputGradient: null,
                    }));
                const outputLayerGradient = MatrixOps.zeros(
                    this.embeddingDim,
                    this.vocabSize
                );

                // 샘플별 순전파 및 역전파
                for (const sample of batch) {
                    // 순전파 - 예측 생성
                    const probabilities = this.forward(sample.input);

                    // 손실 계산
                    const loss = Loss.crossEntropy(
                        probabilities,
                        sample.target
                    );
                    batchLoss += loss;

                    // 소프트맥스 출력 그래디언트 계산 (y_pred - y_true)
                    const outputGradient = MatrixOps.copy(probabilities);
                    for (let i = 0; i < outputGradient.length; i++) {
                        outputGradient[i][sample.target[i]] -= 1;
                    }

                    // 역전파 - 그래디언트 계산
                    const gradients = this.backward(outputGradient);

                    // 그래디언트 누적 (배치 평균용)
                    // 임베딩 그래디언트 누적
                    for (let i = 0; i < this.vocabSize; i++) {
                        for (let j = 0; j < this.embeddingDim; j++) {
                            embeddingGradients[i][j] +=
                                gradients.embeddingGradient.weightGradient[i][
                                    j
                                ];
                        }
                    }

                    // 레이어별 그래디언트 누적
                    for (let l = 0; l < this.layers.length; l++) {
                        const layerGradient = gradients.layerGradients[l];

                        // 어텐션 그래디언트 누적
                        for (let h = 0; h < this.numHeads; h++) {
                            const headGradient =
                                layerGradient.attentionGradient.headsGradient[
                                    h
                                ];
                            for (let i = 0; i < this.embeddingDim; i++) {
                                for (
                                    let j = 0;
                                    j < this.embeddingDim / this.numHeads;
                                    j++
                                ) {
                                    layerGradients[
                                        l
                                    ].attentionGradient.headsGradient[
                                        h
                                    ].queryGradient[i][j] +=
                                        headGradient.queryGradient[i][j];
                                    layerGradients[
                                        l
                                    ].attentionGradient.headsGradient[
                                        h
                                    ].keyGradient[i][j] +=
                                        headGradient.keyGradient[i][j];
                                    layerGradients[
                                        l
                                    ].attentionGradient.headsGradient[
                                        h
                                    ].valueGradient[i][j] +=
                                        headGradient.valueGradient[i][j];
                                }
                            }
                        }

                        // 어텐션 출력 가중치 그래디언트 누적
                        for (
                            let i = 0;
                            i <
                            this.numHeads * (this.embeddingDim / this.numHeads);
                            i++
                        ) {
                            for (let j = 0; j < this.embeddingDim; j++) {
                                layerGradients[
                                    l
                                ].attentionGradient.outputWeightGradient[i][
                                    j
                                ] +=
                                    layerGradient.attentionGradient.outputWeightGradient[
                                        i
                                    ][j];
                            }
                        }

                        // 피드포워드 그래디언트 누적
                        for (let i = 0; i < this.embeddingDim; i++) {
                            for (let j = 0; j < this.feedForwardDim; j++) {
                                layerGradients[
                                    l
                                ].feedForwardGradient.weight1Gradient[i][j] +=
                                    layerGradient.feedForwardGradient.weight1Gradient[
                                        i
                                    ][j];
                            }
                        }

                        for (let i = 0; i < this.feedForwardDim; i++) {
                            for (let j = 0; j < this.embeddingDim; j++) {
                                layerGradients[
                                    l
                                ].feedForwardGradient.weight2Gradient[i][j] +=
                                    layerGradient.feedForwardGradient.weight2Gradient[
                                        i
                                    ][j];
                            }
                        }

                        // 레이어 정규화 그래디언트 누적
                        for (let i = 0; i < this.embeddingDim; i++) {
                            layerGradients[l].layerNorm1Gradient.gammaGradient[
                                i
                            ] +=
                                layerGradient.layerNorm1Gradient.gammaGradient[
                                    i
                                ];
                            layerGradients[l].layerNorm1Gradient.betaGradient[
                                i
                            ] +=
                                layerGradient.layerNorm1Gradient.betaGradient[
                                    i
                                ];
                            layerGradients[l].layerNorm2Gradient.gammaGradient[
                                i
                            ] +=
                                layerGradient.layerNorm2Gradient.gammaGradient[
                                    i
                                ];
                            layerGradients[l].layerNorm2Gradient.betaGradient[
                                i
                            ] +=
                                layerGradient.layerNorm2Gradient.betaGradient[
                                    i
                                ];
                        }
                    }

                    // 출력 레이어 그래디언트 누적
                    for (let i = 0; i < this.embeddingDim; i++) {
                        for (let j = 0; j < this.vocabSize; j++) {
                            outputLayerGradient[i][j] +=
                                gradients.outputLayerGradient[i][j];
                        }
                    }
                }

                // 배치 크기로 그래디언트 스케일링 (평균)
                const batchScale = 1 / batch.length;

                // 임베딩 그래디언트 스케일링
                for (let i = 0; i < this.vocabSize; i++) {
                    for (let j = 0; j < this.embeddingDim; j++) {
                        embeddingGradients[i][j] *= batchScale;
                    }
                }

                // 레이어 그래디언트 스케일링
                for (let l = 0; l < this.layers.length; l++) {
                    // 어텐션 그래디언트 스케일링
                    for (let h = 0; h < this.numHeads; h++) {
                        for (let i = 0; i < this.embeddingDim; i++) {
                            for (
                                let j = 0;
                                j < this.embeddingDim / this.numHeads;
                                j++
                            ) {
                                layerGradients[
                                    l
                                ].attentionGradient.headsGradient[
                                    h
                                ].queryGradient[i][j] *= batchScale;
                                layerGradients[
                                    l
                                ].attentionGradient.headsGradient[
                                    h
                                ].keyGradient[i][j] *= batchScale;
                                layerGradients[
                                    l
                                ].attentionGradient.headsGradient[
                                    h
                                ].valueGradient[i][j] *= batchScale;
                            }
                        }
                    }

                    // 어텐션 출력 가중치 그래디언트 스케일링
                    for (
                        let i = 0;
                        i < this.numHeads * (this.embeddingDim / this.numHeads);
                        i++
                    ) {
                        for (let j = 0; j < this.embeddingDim; j++) {
                            layerGradients[
                                l
                            ].attentionGradient.outputWeightGradient[i][j] *=
                                batchScale;
                        }
                    }

                    // 피드포워드 그래디언트 스케일링
                    for (let i = 0; i < this.embeddingDim; i++) {
                        for (let j = 0; j < this.feedForwardDim; j++) {
                            layerGradients[
                                l
                            ].feedForwardGradient.weight1Gradient[i][j] *=
                                batchScale;
                        }
                    }

                    for (let i = 0; i < this.feedForwardDim; i++) {
                        for (let j = 0; j < this.embeddingDim; j++) {
                            layerGradients[
                                l
                            ].feedForwardGradient.weight2Gradient[i][j] *=
                                batchScale;
                        }
                    }

                    // 레이어 정규화 그래디언트 스케일링
                    for (let i = 0; i < this.embeddingDim; i++) {
                        layerGradients[l].layerNorm1Gradient.gammaGradient[i] *=
                            batchScale;
                        layerGradients[l].layerNorm1Gradient.betaGradient[i] *=
                            batchScale;
                        layerGradients[l].layerNorm2Gradient.gammaGradient[i] *=
                            batchScale;
                        layerGradients[l].layerNorm2Gradient.betaGradient[i] *=
                            batchScale;
                    }
                }

                // 출력 레이어 그래디언트 스케일링
                for (let i = 0; i < this.embeddingDim; i++) {
                    for (let j = 0; j < this.vocabSize; j++) {
                        outputLayerGradient[i][j] *= batchScale;
                    }
                }

                // 파라미터 업데이트 (Adam 옵티마이저)
                this.updateParameters(
                    {
                        embeddingGradient: {
                            weightGradient: embeddingGradients,
                        },
                        layerGradients,
                        outputLayerGradient,
                    },
                    learningRate
                );

                // 배치 손실 평균 계산
                batchLoss /= batch.length;
                totalLoss += batchLoss;
                batchCount++;

                // 진행 상황 출력 (10% 간격, 로깅 빈도 제한)
                if (
                    batchCount %
                        Math.max(
                            1,
                            Math.floor(trainingData.length / batchSize / 10)
                        ) ===
                    0
                ) {
                    const progress = Math.min(
                        100,
                        Math.round((batchStart / trainingData.length) * 100)
                    );
                    console.log(
                        `epoch ${
                            epoch + 1
                        }/${epochs}, percentage: ${progress}%, batch: ${batchLoss.toFixed(
                            4
                        )}`
                    );
                }
            }

            // 에폭 완료, 평균 손실 계산 및 출력
            const avgLoss = totalLoss / batchCount;
            const endTime = Date.now();
            console.log(
                `epoch ${epoch + 1}/${epochs} finished, loss: ${avgLoss.toFixed(
                    4
                )}, ms: ${((endTime - startTime) / 1000).toFixed(2)} seconds`
            );

            // 중간 평가 - 샘플 텍스트 생성
            if ((epoch + 1) % 5 === 0 || epoch === epochs - 1) {
                const demoPrompt = "안녕하세요";
                const generatedText = this.generate(demoPrompt, 5, 0.8);
                console.log(`sample (prompt: ${demoPrompt}): ${generatedText}`);
            }
        }

        console.log("learning completed.");
    }

    // 그래디언트 누적 - 배치 평균 계산용
    private accumulateGradients(
        outputGrad: Tensor2D,
        gradients: Record<string, Tensor2D>
    ): void {
        // 출력 레이어 그래디언트 초기화
        if (!gradients["outputLayer"]) {
            gradients["outputLayer"] = MatrixOps.scale(this.outputLayer, 0);
        }

        // 여기서 각 파라미터에 대한 그래디언트 누적 로직 추가 가능
        // (실제 구현은 updateParameters에서 처리)
    }

    // 파라미터 업데이트 - Adam 옵티마이저 적용
    private updateParameters(
        gradients: {
            embeddingGradient: ReturnType<Embedding["backward"]>;
            layerGradients: ReturnType<TransformerLayer["backward"]>[];
            outputLayerGradient: Tensor2D;
        },
        learningRate: number = 0.001
    ): void {
        // 학습률 설정
        this.optimizer = new AdamOptimizer(learningRate);

        // 1. 임베딩 가중치 업데이트
        const embeddingParams = this.embedding.getParams();
        const updatedEmbeddingWeight = this.optimizer.update(
            "embedding",
            embeddingParams.weight,
            gradients.embeddingGradient.weightGradient
        );
        this.embedding.setParams({
            ...embeddingParams,
            weight: updatedEmbeddingWeight,
        });

        // 2. 각 트랜스포머 레이어 파라미터 업데이트
        for (let i = 0; i < this.layers.length; i++) {
            const layerGradient = gradients.layerGradients[i];
            const layerParams = this.layers[i].getParams();

            // 2.1. 어텐션 파라미터 업데이트
            const attentionGradient = layerGradient.attentionGradient;
            const attentionParams = layerParams.attention;

            // 헤드별 파라미터 업데이트
            const updatedHeads = attentionParams.heads.map(
                (head, headIndex) => {
                    const headGradient =
                        attentionGradient.headsGradient[headIndex];
                    return {
                        queryWeight: this.optimizer.update(
                            `layer${i}_head${headIndex}_query`,
                            head.queryWeight,
                            headGradient.queryGradient
                        ),
                        keyWeight: this.optimizer.update(
                            `layer${i}_head${headIndex}_key`,
                            head.keyWeight,
                            headGradient.keyGradient
                        ),
                        valueWeight: this.optimizer.update(
                            `layer${i}_head${headIndex}_value`,
                            head.valueWeight,
                            headGradient.valueGradient
                        ),
                    };
                }
            );

            // 어텐션 출력 가중치 업데이트
            const updatedOutputWeight = this.optimizer.update(
                `layer${i}_attention_output`,
                attentionParams.outputWeight,
                attentionGradient.outputWeightGradient
            );

            // 2.2. 피드포워드 파라미터 업데이트
            const feedForwardGradient = layerGradient.feedForwardGradient;
            const feedForwardParams = layerParams.feedForward;

            const updatedWeight1 = this.optimizer.update(
                `layer${i}_feedforward_weight1`,
                feedForwardParams.weight1,
                feedForwardGradient.weight1Gradient
            );

            const updatedWeight2 = this.optimizer.update(
                `layer${i}_feedforward_weight2`,
                feedForwardParams.weight2,
                feedForwardGradient.weight2Gradient
            );

            // 2.3. 레이어 정규화 파라미터 업데이트 (간소화된 방식)
            const layerNorm1Gradient = layerGradient.layerNorm1Gradient;
            const layerNorm1Params = layerParams.layerNorm1;

            const updatedGamma1 = new Array(this.embeddingDim);
            const updatedBeta1 = new Array(this.embeddingDim);

            for (let j = 0; j < this.embeddingDim; j++) {
                updatedGamma1[j] =
                    layerNorm1Params.gamma[j] -
                    learningRate * layerNorm1Gradient.gammaGradient[j];
                updatedBeta1[j] =
                    layerNorm1Params.beta[j] -
                    learningRate * layerNorm1Gradient.betaGradient[j];
            }

            const layerNorm2Gradient = layerGradient.layerNorm2Gradient;
            const layerNorm2Params = layerParams.layerNorm2;

            const updatedGamma2 = new Array(this.embeddingDim);
            const updatedBeta2 = new Array(this.embeddingDim);

            for (let j = 0; j < this.embeddingDim; j++) {
                updatedGamma2[j] =
                    layerNorm2Params.gamma[j] -
                    learningRate * layerNorm2Gradient.gammaGradient[j];
                updatedBeta2[j] =
                    layerNorm2Params.beta[j] -
                    learningRate * layerNorm2Gradient.betaGradient[j];
            }

            // 2.4. 레이어 파라미터 일괄 업데이트
            this.layers[i].setParams({
                attention: {
                    heads: updatedHeads,
                    outputWeight: updatedOutputWeight,
                },
                feedForward: {
                    weight1: updatedWeight1,
                    weight2: updatedWeight2,
                },
                layerNorm1: {
                    gamma: updatedGamma1,
                    beta: updatedBeta1,
                },
                layerNorm2: {
                    gamma: updatedGamma2,
                    beta: updatedBeta2,
                },
            });
        }

        // 3. 출력 레이어 가중치 업데이트
        this.outputLayer = this.optimizer.update(
            "outputLayer",
            this.outputLayer,
            gradients.outputLayerGradient
        );
    }

    // 배열 셔플링 - Fisher-Yates 알고리즘
    private shuffle<T>(array: T[]): void {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    // 텍스트 생성 - 자기회귀적 샘플링
    public generate(
        prompt: string, // 시작 프롬프트
        maxLength: number = 50, // 최대 생성 길이
        temperature: number = 1.0 // 샘플링 온도 (높을수록 다양성 증가)
    ): string {
        // 프롬프트 토큰화
        const indices = this.embedding.textToIndices(prompt);
        const generatedIndices = [...indices];

        // 토큰 시퀀셜 생성
        for (let i = 0; i < maxLength; i++) {
            // 컨텍스트 기반 다음 토큰 예측
            const nextTokenProbs = this.forward(
                generatedIndices.slice(-this.maxSeqLength)
            );
            const lastTokenProbs = nextTokenProbs[nextTokenProbs.length - 1];

            // UNK 토큰 확률 제거 (생성 품질 향상)
            const unkIndex = this.vocabSize - 1;
            lastTokenProbs[unkIndex] = 0;

            // 온도 스케일링 적용 (샘플링 엔트로피 조절)
            const scaledProbs = lastTokenProbs.map((p) =>
                Math.exp(Math.log(Math.max(p, 1e-10)) / temperature)
            );
            const sumProbs = scaledProbs.reduce((a, b) => a + b, 0);
            const normalizedProbs = scaledProbs.map((p) => p / sumProbs);

            // Top-K 샘플링 (다양성과 품질 균형)
            const topK = 5;
            const topIndices = this.getTopK(normalizedProbs, topK);

            // UNK 토큰 필터링
            if (topIndices.includes(unkIndex)) {
                console.warn(
                    "UNK 토큰이 Top-K에 포함되어 있습니다. 제외 후 재샘플링."
                );
                const filteredIndices = topIndices.filter(
                    (idx) => idx !== unkIndex
                );
                if (filteredIndices.length === 0) {
                    console.error("샘플링할 토큰이 없습니다. 임의 선택.");
                    const randomToken = Math.floor(
                        Math.random() * (this.vocabSize - 1)
                    ); // UNK 제외
                    generatedIndices.push(randomToken);
                    continue;
                }
                const nextToken = this.sampleFromTopK(
                    normalizedProbs,
                    filteredIndices
                );
                generatedIndices.push(nextToken);
            } else {
                const nextToken = this.sampleFromTopK(
                    normalizedProbs,
                    topIndices
                );
                generatedIndices.push(nextToken);
            }
        }

        // 생성된 토큰 시퀀스를 텍스트로 변환 (프롬프트 제외)
        return this.embedding.indicesToText(
            generatedIndices.slice(indices.length)
        );
    }

    // Top-K 인덱스 선택 - 높은 확률 토큰 필터링
    private getTopK(probs: Tensor1D, k: number): number[] {
        const unkIndex = this.vocabSize - 1;
        return probs
            .map((p, i) => ({ p, i }))
            .filter((item) => item.i !== unkIndex) // UNK 제외
            .sort((a, b) => b.p - a.p) // 확률 내림차순 정렬
            .slice(0, k) // 상위 K개 선택
            .map((item) => item.i); // 인덱스만 추출
    }

    // Top-K 샘플링 - 상위 토큰 중 확률적 선택
    private sampleFromTopK(probs: Tensor1D, topIndices: number[]): number {
        // Top-K 집합 내 확률 정규화
        const filteredProbs = new Array(probs.length).fill(0);
        let sum = 0;
        for (const idx of topIndices) {
            filteredProbs[idx] = probs[idx];
            sum += probs[idx];
        }

        // 확률 재정규화 (합이 1이 되도록)
        for (let i = 0; i < filteredProbs.length; i++) {
            if (filteredProbs[i] > 0) {
                filteredProbs[i] /= sum;
            }
        }

        // 샘플링 실행
        return this.sampleFromDistribution(filteredProbs);
    }

    // 확률 분포에서 샘플링 - 누적 합 기법
    private sampleFromDistribution(probs: Tensor1D): number {
        const rand = Math.random();
        let cumSum = 0;

        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (rand < cumSum) {
                return i;
            }
        }

        return probs.length - 1; // 기본 폴백
    }

    // 순전파 - 전체 트랜스포머 모델 계산
    private forward(indices: number[]): Tensor2D {
        this.lastIndices = [...indices];

        // 1. 토큰 임베딩
        this.lastEmbedding = this.embedding.forward(indices);

        // 2. 위치 인코딩 추가
        this.lastPositionalEncoding =
            this.positionalEncoding.addPositionalEncoding(this.lastEmbedding);

        // 3. 트랜스포머 레이어 스택 통과
        this.lastLayerOutputs = [];
        let output = this.lastPositionalEncoding;
        for (let i = 0; i < this.layers.length; i++) {
            output = this.layers[i].forward(output);
            this.lastLayerOutputs.push(MatrixOps.copy(output));
        }

        // 4. 출력 투영 (언어 모델링 헤드)
        this.lastLogits = MatrixOps.multiply(output, this.outputLayer);

        // 5. 소프트맥스 적용 (확률 분포)
        this.lastProbabilities = this.lastLogits.map((row) =>
            MatrixOps.softmax(row)
        );

        return this.lastProbabilities;
    }

    // 역전파 - 그래디언트 계산
    private backward(outputGradient: Tensor2D): {
        embeddingGradient: ReturnType<Embedding["backward"]>;
        layerGradients: ReturnType<TransformerLayer["backward"]>[];
        outputLayerGradient: Tensor2D;
    } {
        if (
            !this.lastIndices ||
            !this.lastEmbedding ||
            !this.lastPositionalEncoding ||
            !this.lastLayerOutputs ||
            !this.lastLogits ||
            !this.lastProbabilities
        ) {
            throw new Error("순전파가 먼저 실행되어야 합니다.");
        }

        // 1. 출력 레이어 그래디언트 (∂L/∂W_out = lastOutput^T · ∂L/∂logits)
        const outputLayerGradient = MatrixOps.multiply(
            MatrixOps.transpose(
                this.lastLayerOutputs[this.lastLayerOutputs.length - 1]
            ),
            outputGradient
        );

        // 2. 마지막 레이어 출력에 대한 그래디언트 (∂L/∂lastOutput = ∂L/∂logits · W_out^T)
        let layerOutputGradient = MatrixOps.multiply(
            outputGradient,
            MatrixOps.transpose(this.outputLayer)
        );

        // 3. 레이어 스택 역전파 (역순)
        const layerGradients = [];
        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layerGradient = this.layers[i].backward(layerOutputGradient);
            layerGradients.unshift(layerGradient); // 역순 삽입
            layerOutputGradient = layerGradient.inputGradient;
        }

        // 4. 포지셔널 인코딩 역전파
        const positionalEncodingGradient =
            this.positionalEncoding.backward(layerOutputGradient);

        // 5. 임베딩 역전파
        const embeddingGradient = this.embedding.backward(
            positionalEncodingGradient.inputGradient
        );

        return {
            embeddingGradient,
            layerGradients,
            outputLayerGradient,
        };
    }

    // 모델 저장 - JSON 직렬화
    public save(): string {
        const modelData = {
            name: this.name,
            vocabSize: this.vocabSize,
            embeddingDim: this.embeddingDim,
            numLayers: this.numLayers,
            numHeads: this.numHeads,
            feedForwardDim: this.feedForwardDim,
            maxSeqLength: this.maxSeqLength,
            embedding: this.embedding.getParams(),
            positionalEncoding: this.positionalEncoding.getParams(),
            layers: this.layers.map((layer) => layer.getParams()),
            outputLayer: this.outputLayer,
        };

        return JSON.stringify(modelData);
    }

    // 모델 로드 - 정적 팩토리 메서드
    public static load(jsonStr: string): Transformer {
        const data = JSON.parse(jsonStr);

        // 모델 인스턴스 생성
        const model = new Transformer({
            name: data.name,
            vocabSize: data.vocabSize,
            embeddingDim: data.embeddingDim,
            numLayers: data.numLayers,
            numHeads: data.numHeads,
            feedForwardDim: data.feedForwardDim,
            maxSeqLength: data.maxSeqLength,
        });

        // 모델 파라미터 복원
        model.embedding.setParams(data.embedding);
        model.positionalEncoding.setParams(data.positionalEncoding);

        for (let i = 0; i < model.layers.length; i++) {
            model.layers[i].setParams(data.layers[i]);
        }

        model.outputLayer = data.outputLayer;

        return model;
    }
}

// 모델 학습 및 저장 - 실행 부분
const trainAndSaveModel = async (): Promise<void> => {
    const model = new Transformer({
        name: "whatlm_v0-4",
        vocabSize: 2000, // 한국어 어휘 크기
        embeddingDim: 128, // 임베딩 차원
        numLayers: 3, // 트랜스포머 레이어 수
        numHeads: 4, // 어텐션 헤드 수
        feedForwardDim: 256, // 피드포워드 차원 (임베딩의 2배)
        maxSeqLength: 64, // 최대 시퀀스 길이
    });

    // 학습 데이터 (인사말 바리에이션)
    const trainingTexts = [
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
    ];

    // 모델 학습 실행
    console.log("training natural language...");
    model.trainNL(trainingTexts, 30, 4, 0.0005);

    // 학습된 모델 저장
    const savedModel = model.save();
    await fs.writeFile("./models/v0-4/model.json", savedModel);
    console.log("model saved.");
};

// 모델 로드 및 실행
const loadAndRunModel = async (): Promise<void> => {
    // 저장된 모델 로드
    const loadedModel = Transformer.load(
        await fs.readFile("./models/v0-4/model.json", "utf-8")
    );

    // 테스트 프롬프트로 텍스트 생성
    const prompts = ["오늘의"];

    for (const prompt of prompts) {
        console.log(`prompt: ${prompt}`);

        // 다양한 온도 설정으로 생성 테스트
        console.log(`whatlm: ${loadedModel.generate(prompt, 15, 0.8)}\n`);
    }
};

// 실행 코드
// trainAndSaveModel();
loadAndRunModel();
