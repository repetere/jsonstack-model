//@ts-nocheck
import { TextEmbedding, setBackend } from './index';
import * as tf from '@tensorflow/tfjs-node';

let housingDataCSV;
let DataSet;
setBackend(tf);


/** @test {TextEmbedding} */
describe('TextEmbedding', function () {
  beforeAll(async function () {
    // housingDataCSV = await ms.csv.loadCSV('./test/mock/data/boston_housing_data.csv');
    // DataSet = new ms.DataSet(housingDataCSV);
    // DataSet.fitColumns({
    //   columns: columns.map(scaleColumnMap),
    //   returnData:false,
    // });
    return true;
  });
  /** @test {TextEmbedding#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const TE = new TextEmbedding();
      //@ts-expect-error
      const TEConfigured = new TextEmbedding({ test: 'prop', });
      expect(typeof TextEmbedding).toBe('function');
      expect(TE).toBeInstanceOf(TextEmbedding);
      //@ts-expect-error
      expect(TEConfigured.settings.test).toEqual('prop');
    });
  });
  /** @test {TextEmbedding#train} */
  describe('train', () => {
    it('should Load and Return Universal Sentence Encoder and Tokenizer', async function () {
      const NN = new TextEmbedding();
      const trainedModel = await NN.train();
      expect(typeof trainedModel).toBe('object');
      // expect(trainedModel).to.be.instanceOf(UniversalSentenceEncoder);
      // expect(trainedModel2).to.be.an('object');
    },10000);
  });
  /** @test {TextEmbedding#calculate} */
  describe('calculate', () => {
    it('should throw an error if input is invalid', () => {
      const NN = new TextEmbedding();
      expect(typeof NN.calculate).toBe('function');
      //@ts-expect-error
      expect(NN.calculate.bind()).toThrowError(/invalid input array of sentences/);
      expect(NN.calculate.bind(null, 'invalid')).toThrowError(/invalid input array of sentences/);
    },10000);
    it('should train a TextEmbedder', async function () {
      const TextEmbedder = new TextEmbedding();
      await TextEmbedder.train();
      const sentences = [
        'Hello.',
        'How are you?',
      ];
      const predictions = await TextEmbedder.predict(sentences);
      const tokens = await TextEmbedder.tokenizer.encode('Hello, how are you?');
      expect(tokens).toBeInstanceOf(Array);
      expect(tokens).toEqual(expect.arrayContaining([341, 4125, 8, 140, 31, 19, 54, ]));
      expect(predictions).toBeInstanceOf(Array);
      expect(predictions).toHaveLength(2);
      expect(predictions[0]).toHaveLength(512);
    },10000);
    it('should handle empty text', async function () {
      const TextEmbedder = new TextEmbedding();
      await TextEmbedder.train();
      const sentences = [
        ' ',
      ];
      const predictions = await TextEmbedder.predict(sentences);
      // console.log('predictions[0]', predictions[0]);
      const tokens = await TextEmbedder.tokenizer.encode('Hello, how are you?');
      // console.log('tokens',tokens)
      expect(Array.isArray(predictions)).toBe(true)
      expect(predictions.length).toBe(sentences.length)
      //@ts-expect-error
      expect(predictions[0].length).toBe(512);
    }, 10000);
  });
});