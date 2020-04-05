import { TensorScriptModelInterface, TensorScriptOptions, TensorScriptProperties, Matrix, Vector, PredictionOptions, InputTextArray } from './model_interface';
/**
 * Text Embedding with Tensorflow Universal Sentence Encoder (USE)
 * @class TextEmbedding
 * @implements {TensorScriptModelInterface}
 */
export declare class TextEmbedding extends TensorScriptModelInterface {
    /**
     * @param {Object} options - Options for USE
     * @param {{model:Object,tf:Object,}} properties - extra instance properties
     */
    constructor(options?: TensorScriptOptions, properties?: TensorScriptProperties);
    /**
     * Asynchronously loads Universal Sentence Encoder and tokenizer
     * @override
     * @return {Object} returns loaded UniversalSentenceEncoder model
     */
    train(): Promise<any>;
    /**
     * Calculates sentence embeddings
     * @override
     * @param {Array<Array<number>>|Array<number>} input_array - new test independent variables
     * @param {Object} options - model prediction options
     * @return {{data: Promise}} returns tensorflow prediction
     */
    calculate(input_array: InputTextArray, options?: {}): any;
    /**
     * Returns prediction values from tensorflow model
     * @param {Array<string>} input_matrix - array of sentences to embed
     * @param {Boolean} [options.json=true] - return object instead of typed array
     * @param {Boolean} [options.probability=true] - return real values instead of integers
     * @return {Array<Array<number>>} predicted model values
     */
    predict(input_array: InputTextArray, options?: PredictionOptions): Promise<Matrix | Vector>;
}
