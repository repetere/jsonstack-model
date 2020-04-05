import chai from 'chai';
import path from 'path';
import * as ms from '@modelx/data';
import sinonChai from 'sinon-chai';
import chaiAsPromised from 'chai-as-promised';
import { TextEmbedding, } from './index';

const expect = chai.expect;
let housingDataCSV;
let DataSet;

chai.use(sinonChai);
chai.use(chaiAsPromised);

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
      const TEConfigured = new TextEmbedding({ test: 'prop', });
      expect(TextEmbedding).to.be.a('function');
      expect(TE).to.be.instanceOf(TextEmbedding);
      expect(TEConfigured.settings.test).to.eql('prop');
    });
  });
  /** @test {TextEmbedding#train} */
  describe('train', () => {
    it('should Load and Return Universal Sentence Encoder and Tokenizer', async function () {
      const NN = new TextEmbedding();
      const trainedModel = await NN.train();
      expect(trainedModel).to.be.an('object');
      // expect(trainedModel).to.be.instanceOf(UniversalSentenceEncoder);
      // expect(trainedModel2).to.be.an('object');
    },10000);
  });
  /** @test {TextEmbedding#calculate} */
  describe('calculate', () => {
    it('should throw an error if input is invalid', () => {
      const NN = new TextEmbedding();
      expect(NN.calculate).to.be.a('function');
      expect(NN.calculate.bind()).to.throw(/invalid input array of sentences/);
      expect(NN.calculate.bind(null, 'invalid')).to.throw(/invalid input array of sentences/);
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
      expect(tokens).to.be.an('array').that.includes.members([341, 4125, 8, 140, 31, 19, 54, ]);
      expect(predictions).to.be.an('array');
      expect(predictions).to.have.lengthOf(2);
      expect(predictions[0]).to.have.lengthOf(512);
    },10000);
    it('should handle empty text', async function () {
      const TextEmbedder = new TextEmbedding();
      await TextEmbedder.train();
      const sentences = [
        '',
      ];
      const predictions = await TextEmbedder.predict(sentences);
      // console.log('predictions[0]', predictions[0]);
      // const tokens = await TextEmbedder.tokenizer.encode('Hello, how are you?');
      // expect(predictions).to.be.an('array');
      // expect(predictions).to.have.lengthOf(2);
      // expect(predictions[0]).to.have.lengthOf(512);
    }, 10000);
  });
});