//@ts-nocheck
import path from 'path';
import * as ms from '@jsonstack/data';
import * as tf from '@tensorflow/tfjs-node';
import { BaseNeuralNetwork, setBackend } from './index';
const independentVariables = [
  'CRIM',
  'ZN',
  'INDUS',
  'CHAS',
  'NOX',
  'RM',
  'AGE',
  'DIS',
  'RAD',
  'TAX',
  'PTRATIO',
  'B',
  'LSTAT',
];
const dependentVariables = [
  'MEDV',
];
const columns = independentVariables.concat(dependentVariables);
let housingDataCSV;
let DataSet;

function scaleColumnMap(columnName) {
  return {
    name: columnName,
    options: {
      strategy: 'scale',
      scaleOptions: {
        strategy:'standard',
      },
    },
  };
}

setBackend(tf);

/** @test {BaseNeuralNetwork} */
describe('BaseNeuralNetwork', function () {
  beforeAll(async function () {
    const fpath = `${path.join(__dirname, '/test/mock/data/boston_housing_data.csv')}`;

    housingDataCSV = await ms.csv.loadCSV(fpath);
    DataSet = new ms.DataSet(housingDataCSV);
    DataSet.fitColumns({
      columns: columns.map(scaleColumnMap),
      returnData:false,
    });
  });
  /** @test {BaseNeuralNetwork#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const MLR = new BaseNeuralNetwork();
      //@ts-ignore
      const MLRConfigured = new BaseNeuralNetwork({ test: 'prop', });
      expect(typeof BaseNeuralNetwork).toBe('function');
      expect(MLR).toBeInstanceOf(BaseNeuralNetwork);
      //@ts-ignore
      expect(MLRConfigured.settings.test).toEqual('prop');
    });
  });
  /** @test {BaseNeuralNetwork#generateLayers} */
  describe('generateLayers', () => {
    it('should throw an error if generateLayers method is not implemented', () => {
      class NN extends BaseNeuralNetwork{
        generateLayers(x, y, layers) {
          return true;
        }
      }
      const TS = new BaseNeuralNetwork();
      const TSNN = new NN();
      expect(typeof TS.generateLayers).toBe('function');
      expect(TS.generateLayers.bind(null)).toThrowError('generateLayers method is not implemented');
      expect(typeof TSNN.generateLayers).toBe('function');
      expect(TSNN.generateLayers.bind(null)).toBeTruthy();
    });
  });
  /** @test {BaseNeuralNetwork#train} */
  describe('train', () => {
    it('should train a NN', async function () {
      const NN = new BaseNeuralNetwork();
      const x = [];
      const y = [];
      const layers = [];
      const tf = {
        tensor: () => ({ 
          dispose: () => { },
        }),
        sequential: () => ({
          compile: () => true,
          fit: () => true,
        }),
      };
      const settings = {};
      function getInputShape() { }
      function generateLayers() { }
      const trainedModel = await NN.train.call({
        getInputShape,
        generateLayers,
        tf,
        settings,
      }, x, y, layers);
      const trainedModel2 = await NN.train.call({
        getInputShape,
        generateLayers,
        tf,
        settings,
        layers:[],
      }, x, y);
      expect(typeof trainedModel).toBe('object');
      expect(typeof trainedModel2).toBe('object');
    });
  });
  /** @test {BaseNeuralNetwork#calculate} */
  describe('calculate', () => {
    it('should throw an error if input is invalid', () => {
      const NN = new BaseNeuralNetwork();
      expect(typeof NN.calculate).toBe('function');
      //@ts-ignore
      expect(NN.calculate.bind()).toThrowError(/invalid input matrix/);
      expect(NN.calculate.bind(null, 'invalid')).toThrowError(/invalid input matrix/);
    });
    it('should train a NN', async function () {
      const NN = new BaseNeuralNetwork();
      const x = [1, 2, 3, ];
      const x2 = [[1, 2, 3, ], [1, 2, 3, ], ];
      const tf = {
        tensor: () => ({ 
          dispose: () => { },
        }),
        sequential: () => ({
          compile: () => true,
          fit: () => true,
        }),
      };
      const model = {
        predict: () => true,
      };
      const prediction = NN.calculate.call({
        tf,
        model,
      }, x);
      const prediction2 = NN.calculate.call({
        tf,
        model,
      }, x2);
      expect(prediction).toBe(true);
      expect(prediction2).toBe(true);
    });
  });
});