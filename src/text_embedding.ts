import { TensorScriptModelInterface, TensorScriptOptions, TensorScriptProperties, Matrix, Vector, PredictionOptions, InputTextArray, } from './model_interface';
import { Tokenizer, UniversalSentenceEncoder, load } from '@tensorflow-models/universal-sentence-encoder';
const BASE_PATH = 'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder';
import * as tf from '@tensorflow/tfjs-core';

// import {loadTokenizer} from '@tensorflow-models/universal-sentence-encoder/dist/tokenizer/index';
/**
 * Load the Tokenizer for use independently from the UniversalSentenceEncoder.
 *
 * @param {string} pathToVocabulary (optional) Provide a path to the vocabulary file.
 */
export async function loadVocabulary(pathToVocabulary:string) {
  const vocabulary = await tf.util.fetch(pathToVocabulary);
  return vocabulary.json();
}
export async function loadTokenizer() {
  const vocabulary = await (loadVocabulary(`${BASE_PATH}/vocab.json`));
  // console.log('vocabulary',vocabulary)
  const tokenizer = new Tokenizer(vocabulary);
  return tokenizer;
}
let model:UniversalSentenceEncoder;
let tokenizer:Tokenizer;
/**
 * Text Embedding with Tensorflow Universal Sentence Encoder (USE)
 * @class TextEmbedding
 * @implements {TensorScriptModelInterface}
 */
export class TextEmbedding extends TensorScriptModelInterface {
  /**
   * @param {Object} options - Options for USE
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options:TensorScriptOptions = {}, properties?:TensorScriptProperties) {
    const config = Object.assign({
    }, options);
    super(config, properties);
    this.type = 'TextEmbedding';

    return this;
  }
  /**
   * Asynchronously loads Universal Sentence Encoder and tokenizer
   * @override
   * @return {Object} returns loaded UniversalSentenceEncoder model
   */
  async train() {
    const promises:Promise<any>[] = [];
    if (!model) promises.push(load());
    else promises.push(Promise.resolve(model));
    if (!tokenizer) promises.push(loadTokenizer());
    else promises.push(Promise.resolve(tokenizer));
    const USE = await Promise.all(promises);
    if (!model) model = USE[ 0 ];
    if (!tokenizer) tokenizer = USE[ 1 ];
    // console.log({model,tokenizer})
    this.model = model;
    this.tokenizer = tokenizer;
    this.trained = true;
    this.compiled = true;

    return this.model;
  }
  /**
   * Calculates sentence embeddings
   * @override
   * @param {Array<Array<number>>|Array<number>} input_array - new test independent variables
   * @param {Object} options - model prediction options
   * @return {{data: Promise}} returns tensorflow prediction 
   */
  calculate(input_array:InputTextArray, options = {}) {
    if (!input_array || Array.isArray(input_array) === false) throw new Error('invalid input array of sentences');
    const embeddings = this.model.embed(input_array);
    return embeddings;
  }
  /**
   * Returns prediction values from tensorflow model
   * @param {Array<string>} input_matrix - array of sentences to embed 
   * @param {Boolean} [options.json=true] - return object instead of typed array
   * @param {Boolean} [options.probability=true] - return real values instead of integers
   * @return {Array<Array<number>>} predicted model values
   */
  async predict(input_array:InputTextArray, options:PredictionOptions = {}): Promise<Matrix|Vector> {
    const config = Object.assign({
      json: true,
      probability: true,
    }, options);
    const embeddings = await this.calculate(input_array, options);
    const predictions:number[] = await embeddings.data(); 
    if (config.json === false) {
      return predictions;
    } else {
      const shape = [input_array.length, 512, ];
      const predictionValues = (options.probability === false)
        ? Array.from(predictions).map(Math.round)
        : Array.from(predictions);
      return this.reshape(predictionValues, shape);
    }
  }
}