import path from 'path';
import * as ms from '@modelx/data';
import { DeepLearningRegression, } from './index';
import * as tf from '@tensorflow/tfjs-node';
import '@tensorflow/tfjs-node';
// import '@tensorflow/tfjs-backend-wasm';
// import { setWasmPath } from '@tensorflow/tfjs-backend-wasm';
// const wasmpath = `${path.join(__dirname, '../node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm')}`;
// console.log({ wasmpath });
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
let x_matrix;
let y_matrix;
let nnRegressionDeep;
let nnRegressionWide;
let nnRegressionDeepModel;
let nnRegressionWideModel;
const fit = {
  epochs: 10,
  batchSize: 5,
  // verbose:0,
  callbacks: {
    // onTrainBegin: function(logs){
    //   console.log('onTrainBegin', { logs });
    // },
    // onEpochBegin: function(epoch, logs){
    //   console.log('onEpochBegin', { epoch, logs });
    // },
    // onYield: function(epoch, batch, logs){
    //   console.log('onYield', { epoch, batch, logs });
    // }
    /**
onTrainEnd(logs): called when training ends.
onEpochBegin(epoch, logs): called at the start of every epoch.
onEpochEnd(epoch, logs): called at the end of every epoch.
onBatchBegin(batch, logs): called at the start of every batch.
onBatchEnd(batch, logs): called at the end of every batch.
onYield(epoch, batch, logs): called every yieldEvery milliseconds with the current epoch, batch and logs. The logs are the same as in onBatchEnd(). Note that onYield can skip batches or epochs. See also docs for yieldEvery below.
     */
  }
};
const input_x = [
  [-0.41936692921321594, 0.2845482693404666, -1.2866362317172035, -0.272329067679207, -0.1440748547324509, 0.4132629204530747, -0.119894767215809, 0.1400749839795629, -0.981871187861867, -0.6659491794887338, -1.457557967289609, 0.4406158949991029, -1.074498970343932,],
  [-0.41692666996409716, -0.4872401872268264, -0.5927943782429392, -0.272329067679207, -0.7395303607434242, 0.1940823874370036, 0.3668034264326209, 0.5566090495704026, -0.8670244885881488, -0.9863533804386945, -0.3027944997494681, 0.4406158949991029, -0.49195252491856634,],
];

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
/** @test {DeepLearningRegression} */
describe('DeepLearningRegression', function () {
  beforeAll(async function () {
    tf.setBackend('tensorflow')
    console.log(tf.getBackend());
    const fpath = `${path.join(__dirname, '/test/mock/data/boston_housing_data.csv')}`;

      /*
        housingdataCSV = [ 
          { CRIM: 0.00632, ZN: 18, INDUS: 2.31, CHAS: 0, NOX: 0.538, RM: 6.575, AGE: 65.2, DIS: 4.09, RAD: 1, TAX: 296, PTRATIO: 15.3, B: 396.9, LSTAT: 4.98, MEDV: 24 },
          { CRIM: 0.02731, ZN: 0, INDUS: 7.07, CHAS: 0, NOX: 0.469, RM: 6.421, AGE: 78.9, DIS: 4.9671, RAD: 2, TAX: 242, PTRATIO: 17.8, B: 396.9, LSTAT: 9.14, MEDV: 21.6 },
          ...
        ] 
        */
      housingDataCSV = await ms.csv.loadCSV(fpath);
      DataSet = new ms.DataSet(housingDataCSV);
      DataSet.fitColumns({
        columns: columns.map(scaleColumnMap),
        returnData:false,
      });
      x_matrix = DataSet.columnMatrix(independentVariables); 
      y_matrix = DataSet.columnMatrix(dependentVariables);
      /* x_matrix = [
        [ -0.41936692921321594, 0.2845482693404666, -1.2866362317172035, -0.272329067679207, -0.1440748547324509, 0.4132629204530747, -0.119894767215809, 0.1400749839795629, -0.981871187861867, -0.6659491794887338, -1.457557967289609, 0.4406158949991029, -1.074498970343932 ],
        [ -0.41692666996409716, -0.4872401872268264, -0.5927943782429392, -0.272329067679207, -0.7395303607434242, 0.1940823874370036, 0.3668034264326209, 0.5566090495704026, -0.8670244885881488, -0.9863533804386945, -0.3027944997494681, 0.4406158949991029, -0.49195252491856634 ]
        ...
      ]; 
      y_matrix = [
        [ 0.15952778852449556 ],
        [ -0.1014239172731213 ],
        ...
      ] 
      const y_vector = ms.util.pivotVector(y_matrix)[ 0 ];// not used but just illustrative
      y_vector = [ 0.15952778852449556,
        -0.1014239172731213, ... ]
      */

      nnRegressionDeep = new DeepLearningRegression({ layerPreference:'deep', fit, verbose:0 },{tf});
      nnRegressionWide = new DeepLearningRegression({ layerPreference: 'wide', fit, verbose:0,  },{tf});
      const models = await Promise.all([
        nnRegressionDeep.train(x_matrix, y_matrix),
        nnRegressionWide.train(x_matrix, y_matrix),
      ]);
      nnRegressionDeepModel = models[ 0 ];
      nnRegressionWideModel = models[ 1 ];
      return true;

    
  },120000);
  /** @test {DeepLearningRegression#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const NN = new DeepLearningRegression();
      const NNConfigured = new DeepLearningRegression({ test: 'prop', });
      expect(typeof DeepLearningRegression).toBe('function');
      expect(NN).toBeInstanceOf(DeepLearningRegression);
      expect(NNConfigured.settings.test).toEqual('prop');
    });
  });
  /** @test {DeepLearningRegression#generateLayers} */
  describe('generateLayers', () => {
    it('should generate a deep network', async () => {
      const predictions = await nnRegressionDeep.predict(input_x);
      const predictions_unscaled = predictions.map(pred=>DataSet.scalers.get('MEDV').descale(pred[0]));
      const shape = nnRegressionDeep.getInputShape(predictions);
      // console.log('nnRegressionDeep.layers', nnRegressionDeep.layers);
      // console.log({
      //   predictions_unscaled,
      //   predictions,
      //   shape,
      // });
      expect(predictions).toHaveLength(input_x.length);
      expect(nnRegressionDeep.layers).toHaveLength(3);
      expect(shape).toMatchObject([2, 1,]);
      expect(Math.round(predictions_unscaled[ 0 ])).toBeCloseTo(24, 15);
      expect(Math.round(predictions_unscaled[ 0 ])).toBeCloseTo(21, 15);
    });
    it('should generate a wide network', async () => {
      const predictions = await nnRegressionWide.predict(input_x);
      const predictions_unscaled = predictions.map(pred=>DataSet.scalers.get('MEDV').descale(pred[0]));
      const shape = nnRegressionWide.getInputShape(predictions);
      // console.log('nnRegressionWide.layers', nnRegressionWide.layers);
      expect(predictions).toHaveLength(input_x.length);
      expect(nnRegressionWide.layers).toHaveLength(2);
      expect(shape).toMatchObject([2, 1,]);
      expect(Math.round(predictions_unscaled[ 0 ])).toBeCloseTo(24, 15);
      expect(Math.round(predictions_unscaled[ 0 ])).toBeCloseTo(21, 15);
    });
    it('should generate a network from layers', async () => { 
      const nnRegressionCustom = new DeepLearningRegression({ layerPreference:'custom', fit, });
      await nnRegressionCustom.train(x_matrix, y_matrix, nnRegressionWide.layers);
      expect(nnRegressionCustom.layers).toHaveLength(2);
    },120000);
  });
});