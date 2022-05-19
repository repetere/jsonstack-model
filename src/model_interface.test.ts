//@ts-nocheck
import path from 'path';
import fs from 'fs-extra';
import { asyncForEach, LambdaLayer, } from './model_interface';
import * as tf from '@tensorflow/tfjs-node';
import * as ms from '@jsonstack/data';
import { TensorScriptModelInterface, MultipleLinearRegression, setBackend, } from './index';
import { toBeWithinRange, } from './jest.test';
expect.extend({ toBeWithinRange });

describe('asyncForEach', () => {
  it('should interate asynchrnously', async () => {
    const arr = [4, 3, 2, 1];
    const mockCallback = jest.fn(x =>( x * x));
    await asyncForEach(arr, mockCallback);
    
    // The mock function is called twice
    expect(mockCallback.mock.calls.length).toBe(4);

    // The first argument of the first call to the function was 0
    expect(mockCallback.mock.calls[0][0]).toBe(4);

    // The first argument of the second call to the function was 1
    expect(mockCallback.mock.calls[1][0]).toBe(3);

    // The return value of the first call to the function was 42
    expect(mockCallback.mock.results[0].value).toBe(16);

  });
});

describe('LambdaLayer', () => {
  describe('tf.serialization.registerClass', () => {
    it('should registerClass on tf', () => {
      const mockTF = {
        serialization: {
          registerClass: jest.fn(() => true),
        }
      }
      setBackend(tf);
      new TensorScriptModelInterface({}, { tf: mockTF });
      expect(mockTF.serialization.registerClass.mock.calls.length).toBe(1);
    });
  });
  describe('static className', () => {
    it('should return classname', () => {
      expect(LambdaLayer.className).toBe('LambdaLayer');
    });
  });
  describe('constructor', () => {
    it('should configure a new instance', () => {
      const LL = new LambdaLayer({
        lambdaFunction: 'return tf.mean(input,1)',
        lambdaOutputShape: [1],
        name: 'mean layer',
      });
      const nonameLL = new LambdaLayer({
        lambdaFunction: 'return tf.mean(input,1)',
        lambdaOutputShape: [1],
      });
      expect(LL.name).toBe('mean layer');
      expect(typeof nonameLL.name).toBe('string');
    });
  });
  describe('getConfig', () => {
    it('should return layer configuration', () => {
      const LL = new LambdaLayer({
        lambdaFunction: 'return tf.mean(input,1)',
        lambdaOutputShape: [1],
        name: 'mean layer',
      });
      const config = LL.getConfig();
      expect(config.lambdaFunction).toBe('return tf.mean(input,1)');
      expect(config).toHaveProperty('trainable');
      // console.log({ config });
    });
  });
  describe('computeOutputShape', () => {
    it('should return output shape', () => {
      const LL = new LambdaLayer({
        lambdaFunction: 'return tf.mean(input,1)',
        lambdaOutputShape: [2,2],
        name: 'mean layer',
      });
      const LL2 = new LambdaLayer({
        lambdaFunction: 'return tf.mean(input,1)',
        // lambdaOutputShape: [2,2],
        name: 'mean layer',
      });
      //@ts-ignore
      expect(LL.computeOutputShape([2])).toMatchObject([2, 2]);
      //@ts-ignore
      expect(LL2.computeOutputShape([3, 2])).toBe(3);
    });
  });
  describe('call', () => {
    it('should calculate a layer output tensor', () => {
      const LL = new LambdaLayer({
        lambdaFunction: 'return tf.mean(input)',
        lambdaOutputShape: [2, 2],
        name: 'mean layer',
      });
      const output = LL.call(tf.tensor([3, 5, 7]));
      //@ts-expect-error
      const outputVal = output.asScalar().dataSync()[0];
      expect(output).toBeInstanceOf(tf.Tensor);
      //@ts-expect-error
      expect(output.dtype).toBe('float32');
      expect(outputVal).toBe(5);
      // console.log({ output, outputVal });
    });
  })
});

/** @test {TensorScriptModelInterface} */
describe('TensorScriptModelInterface', function () {
  const independentVariables = ['sqft', 'bedrooms',];
  const dependentVariables = ['price',];
  function scaleColumnMap(columnName) {
    return {
      name: columnName,
      options: {
        strategy: 'scale',
        scaleOptions: {
          strategy: 'standard',
        },
      },
    };
  }

  let housingDataCSV;
  let DataSet;
  let x_matrix;
  let y_matrix;
  let input_x;
  const saveFilePath = `${path.join(__dirname, './test/mock_saved_files/mlr_model_saved')}`;
  const saveModelPath = `file://${saveFilePath}`;
  beforeAll(async function () {
    const fpath = `${path.join(__dirname, '/test/mock/data/portland_housing_data.csv')}`;
    housingDataCSV = await ms.csv.loadCSV(fpath);
    DataSet = new ms.DataSet(housingDataCSV);
    DataSet.fitColumns({
      columns: independentVariables.concat(dependentVariables).map(scaleColumnMap),
      returnData: false,
    });
    x_matrix = DataSet.columnMatrix(independentVariables);
    y_matrix = DataSet.columnMatrix(dependentVariables);
    input_x = [
      [
        DataSet.scalers.get('sqft').scale(4215),
        DataSet.scalers.get('bedrooms').scale(4),
      ], //549000
      [
        DataSet.scalers.get('sqft').scale(852),
        DataSet.scalers.get('bedrooms').scale(2),
      ], //179900
    ];
  },120000);
  /** @test {TensorScriptModelInterface#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      tf.setBackend('tensorflow')
      // console.log(tf.getBackend());

      const TSM = new TensorScriptModelInterface({},{tf});
      const TSMConfigured = new 
      //@ts-expect-error
      TensorScriptModelInterface({ test: 'prop', });
      expect(typeof TensorScriptModelInterface).toBe('function');
      expect(TSM).toBeInstanceOf(TensorScriptModelInterface);
      //@ts-expect-error
      expect(TSMConfigured.settings.test).toBe('prop');
    });
  });
  /** @test {TensorScriptModelInterface#reshape} */
  describe('reshape', () => {
    it('should export a static method', () => {
      expect(typeof TensorScriptModelInterface.reshape).toBe('function');
    });
    it('should reshape an array into a matrix', () => {
      const array = [1, 0, 0, 1,];
      const shape = [2, 2,];
      const matrix = [
        [1, 0, ],
        [0, 1, ],
      ];
      const result = TensorScriptModelInterface.reshape(array, shape);
      expect(result).toMatchObject(matrix);
      // expect(TensorScriptModelInterface.reshape.bind(null, array, [1, 2, ])).to.throw(/specified shape/);
    });
    it('should reshape multiple dimensions', () => {
      const array = [1, 1, 0, 1, 1, 0,];
      const shape = [2, 3, 1,];
      const matrix = [
        [[1,], [1,], [0,],],
        [[1,], [1,], [0,],],
      ];
      const result = TensorScriptModelInterface.reshape(array, shape);
      // console.log({ result });
    });
  });
  /** @test {TensorScriptModelInterface#getInputShape} */
  describe('getInputShape', () => {
    it('should export a static method', () => {
      expect(typeof TensorScriptModelInterface.getInputShape).toBe('function');
    });
    it('should return the shape of a matrix', () => {
      const matrix = [
        [1, 0, ],
        [0, 1, ],
      ];
      const matrix2 = [
        [1, 0, ],
        [1, 0, ],
        [1, 0, ],
        [1, 0, ],
        [1, 0, ],
        [1, 0, ],
        [0, 1, ],
      ];
      const matrix3 = [
        [1, 0, 4, 5, ],
        [1, 0, 4, 5, ],
        [1, 0, 4, 5, ],
        [1, 0, 4, 5, ],
        [1, 0, 4, 5, ],
      ];
      const matrix4 = [
        [1, 0, 4, 5, ],
        [1, 0, 4, ],
        [1, 0, 4, 5, ],
      ];
      const matrix5 = [
        [[1,], [1,], [0,],],
        [[1,], [1,], [0,],],
        [[1,], [1,], [0,],],
        [[1,], [1,], [0,],],
        [[1,], [1,], [0,],],
        [[1,], [1,], [0,],],
      ];
      TensorScriptModelInterface.getInputShape(matrix5);
      expect(TensorScriptModelInterface.getInputShape(matrix)).toMatchObject([2, 2, ]);
      expect(TensorScriptModelInterface.getInputShape(matrix2)).toMatchObject([7, 2, ]);
      expect(TensorScriptModelInterface.getInputShape(matrix3)).toMatchObject([5, 4, ]);
      expect(TensorScriptModelInterface.getInputShape.bind(null, matrix4)).toThrowError(/input must have the same length in each row/);
      expect(TensorScriptModelInterface.getInputShape(matrix5)).toMatchObject([6, 3, 1,]);
    });
    it('should throw an error if input is not a matrix', () => {
      //@ts-expect-error
      expect(TensorScriptModelInterface.getInputShape.bind()).toThrowError(/must be a matrix/);
    });
  });
  /** @test {TensorScriptModelInterface#train} */
  describe('train', () => {
    it('should throw an error if train method is not implemented', () => {
      class MLR extends TensorScriptModelInterface{
      //@ts-expect-error
        train(x, y) {
          return true;
        }
      }
      const TSM = new TensorScriptModelInterface();
      const TSMMLR = new MLR();
      expect(typeof TSM.train).toBe('function');
      expect(TSM.train.bind(null)).toThrowError('train method is not implemented');
      expect(typeof TSMMLR.train).toBe('function');
      expect(TSMMLR.train.bind(null)).toBeTruthy();
    });
  });
  /** @test {TensorScriptModelInterface#calculate} */
  describe('calculate', () => {
    it('should throw an error if calculate method is not implemented', () => {
      class MLR extends TensorScriptModelInterface{
        //@ts-expect-error
        calculate(x, y) {
          return true;
        }
      }
      const TSM = new TensorScriptModelInterface();
      const TSMMLR = new MLR();
      expect(typeof TSM.calculate).toBe('function');
      expect(TSM.calculate.bind(null)).toThrowError('calculate method is not implemented');
      expect(typeof TSMMLR.calculate).toBe('function');
      expect(TSMMLR.calculate.bind(null)).toBeTruthy();
    });
  });
  /** @test {TensorScriptModelInterface#predict} */
  describe('predict', () => {
    class MLR extends TensorScriptModelInterface{
      //@ts-expect-error
      calculate(x) {
        this.yShape = [100, 2, ];
        return {
          data: () => new Promise((resolve) => {
            const predictions = new Float32Array([21.41, 31.74, 41.01, 51.53, ]);
            resolve(predictions);
          }),
        };
      }
    }
    it('should throw an error if input is invalid', async function () {
      const TSMMLR = new MLR();
      try {
        const predictPromise = await TSMMLR.predict();
        expect(predictPromise).toBeFalsy();
      } catch (e) {
        expect(e).toBeInstanceOf(Error);
        expect(e.message).toMatch(/invalid input matrix/);
      }
      try {
        const predictPromiseCatch = await TSMMLR.predict([1, ]);
        expect(predictPromiseCatch).toBeFalsy();
      } catch (e2) {
        expect(e2).toBeInstanceOf(Error);
        expect(e2.message).toMatch(/Dimension mismatch/);
      }
    });
    it('should return predictions', async function () {
      const TSMMLR = new MLR();
      const input = [
        [1, 2,],
        [1, 2,],
      ];
      const predictions = await TSMMLR.predict(input);
      const predictionsRounded = await TSMMLR.predict(input, { probability:false, });
      const predictionsRaw = await TSMMLR.predict(input, { json: false, });
      expect(predictions).toHaveLength(2);
      expect(predictionsRaw).toBeInstanceOf(Float32Array);
      predictionsRounded.forEach(predRow => {
        predRow.forEach(pred => {
          expect(Number.isInteger(pred)).toBe(true);
        });
      });
    });
  });
  /** @test {TensorScriptModelInterface#loadModel} */
  describe('loadModel', () => {
    it('should call tensorflow load model and store it', async function () {
      const TSM = new TensorScriptModelInterface({}, {
        tf: {
          loadLayersModel: () => new Promise((resolve) => resolve({
            inputs:[{ shape:[null, 2,], },],
            outputs:[{ shape:[null, 1,], },],
          })),
        },
      });
      //@ts-expect-error
      const loadedModel = await TSM.loadModel();
      expect(typeof loadedModel).toBe('object');
    });
    it('should load a model and make predictions', async function () {
      const loadedFilePath = `${path.join(__dirname, './test/mock_saved_files/mlr_model/model.json')}`;
      const loadedModelPath = `file://${loadedFilePath}`;
      const trainedModelToLoad = new MultipleLinearRegression({
        fit: {
          epochs: 100,
          batchSize: 5,
          verbose:0,
        },
      });
      
      await trainedModelToLoad.loadModel(loadedModelPath);
      const loaded_predictions = await trainedModelToLoad.predict(input_x);

      const scaledInputs = DataSet.reverseColumnMatrix({ vectors: input_x, labels: independentVariables, });
      const scaledPredictions = DataSet.reverseColumnMatrix({ vectors: loaded_predictions, labels: dependentVariables, });
      const descaledPredictions = scaledPredictions.map((val, i) => {
        const input = scaledInputs[ i ];
        const datum = Object.assign({}, input, val);
        return DataSet.inverseTransformObject(datum);
      });
      //@ts-expect-error
      expect(Math.round(descaledPredictions[0].price)).toBeWithinRange(610000, 650000);
      //@ts-expect-error
      expect(Math.round(descaledPredictions[1].price)).toBeWithinRange(180000, 200000);
    },120000);
  });
  describe('saveModel',  ()=> {
    it('should call tensorflow save model and store it', async function () {
      const TSM = new TensorScriptModelInterface({}, {
        model: {
          save: () => new Promise((resolve) => resolve(true)),
        },
      });
      //@ts-expect-error
      const savedModel = await TSM.saveModel();
      expect(savedModel).toBe(true);
    });
    //TODO: Jest save file doesnt work
    it('[WARN: BROKEN WITH JEST] should save a trained model to a file', async function () {
      let savedModelStatus;
      const trainedMLR = new MultipleLinearRegression({
        fit: {
          epochs: 100,
          batchSize: 5,
          verbose:0,
        },
      });
      const trainedMLRModel = await trainedMLR.train(x_matrix, y_matrix);
      const saved_predictions = await trainedMLR.predict(input_x);
      try {
        savedModelStatus = await trainedMLR.saveModel(saveModelPath);
      } catch (e) {
        // console.log(e);
      }
      // console.log({
      //   // trainedMLR, trainedMLRModel,
      //   saveModelPath,
      //   x_matrix, y_matrix, input_x, saved_predictions,
      // });
      // expect(fs.existsSync(saveFilePath)).toBe(true);
      // expect(fs.existsSync(path.join(saveFilePath, 'model.json'))).toBe(true);
      // expect(fs.existsSync(path.join(saveFilePath, 'weights.bin'))).toBe(true);
      expect(trainedMLRModel).toBeTruthy();
      // expect(savedModelStatus).toHaveProperty('modelArtifactsInfo');
      await fs.remove(saveFilePath);
      // MultipleLinearRegression
    },120000);
  });
  describe('exportConfiguration / importConfiguration',  ()=> {
    it('should export configuration from exportConfiguration', function () {
      const TSM = new MultipleLinearRegression({ stateful: true, });
      const config = TSM.exportConfiguration();
      // const savedModel = await TSM.saveModel();
      expect(config.settings.stateful).toBe(true);
      expect(config.trained).toBe(false);
      expect(config.type).toBe('MultipleLinearRegression');
    });
    it('should import configuration from importConfiguration', function () {
      const trainedMLR = new MultipleLinearRegression({
        fit: {
          epochs: 100,
          batchSize: 5,
          verbose:0,
        },
      });
      trainedMLR.importConfiguration({
        type: 'CustomMLR',
        settings: {},
        compiled: false,
        trained: false,
      });
      // console.log({ trainedMLR });
      expect(trainedMLR.type).toBe('CustomMLR');

      // MultipleLinearRegression
    });
  });
});

