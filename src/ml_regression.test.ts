//@ts-nocheck
import path from 'path';
import * as ms from '@jsonstack/data';
import * as tf from '@tensorflow/tfjs-node';
import * as scikit from 'scikitjs/src/index';
import { MachineLearningLinearRegression, setBackend, setScikit, } from './index';
import { toBeWithinRange, } from './jest.test';
expect.extend({ toBeWithinRange });
setBackend(tf);
scikit.setBackend(tf);
setScikit(scikit);

const independentVariables = ['sqft', 'bedrooms', ];
const dependentVariables = ['price',];
let housingDataCSV;
let input_x;
let DataSet;
let x_matrix;
let y_matrix;
let trainedMLR;
let trainedWithCallbacksMLR;
let trainedCallbackMLR;
let trainedMLRModel;

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
/** @test {MachineLearningLinearRegression} */
describe('MachineLearningLinearRegression', function () {
  beforeAll(async function () {
    console.log = jest.fn()
    const fpath = `${path.join(__dirname, '/test/mock/data/portland_housing_data.csv')}`;

    housingDataCSV = await ms.csv.loadCSV(fpath);
    /*
    housingdataCSV = [ 
      { sqft: 2104, bedrooms: 3, price: 399900 },
      { sqft: 1600, bedrooms: 3, price: 329900 },
      ...
      { sqft: 1203, bedrooms: 3, price: 239500 } 
    ] 
    */
    DataSet = new ms.DataSet(housingDataCSV);
    DataSet.fitColumns({
      columns: independentVariables.concat(dependentVariables).map(scaleColumnMap),
      returnData:false,
    });
    x_matrix = DataSet.columnMatrix(independentVariables); 
    y_matrix = DataSet.columnMatrix(dependentVariables);
    // const y_vector = ms.util.pivotVector(y_matrix)[ 0 ];// not used but just illustrative
    /* x_matrix = [
        [2014, 3],
        [1600, 3],
      ] 
      y_matrix = [
        [399900],
        [329900],
      ] 
      y_vector = [ 399900, 329900]
    */
    trainedMLR = new MachineLearningLinearRegression({
      fit: {
        epochs: 100,
        batchSize: 5,
      },
    });
    trainedMLRModel = await trainedMLR.train(x_matrix, y_matrix);

    trainedWithCallbacksMLR = new MachineLearningLinearRegression({
      fit:{
        callbacks:{
          onEpochBegin: function(epoch:number, logs:unknown){
            console.log('onEpochBegin', { epoch, logs });
          }
        }
      }
    });
    trainedCallbackMLR = await trainedWithCallbacksMLR.train(x_matrix, y_matrix);

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
    return true;
  }, 120000);
  /** @test {MachineLearningLinearRegression#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const MLR = new MachineLearningLinearRegression({
        fit: {
          epochs: 200,
          batchSize: 5,
        },
      });
      //@ts-expect-error
      const MLRConfigured = new MachineLearningLinearRegression({ test: 'prop', }, {});
      expect(typeof MachineLearningLinearRegression).toBe('function');
      expect(MLR).toBeInstanceOf(MachineLearningLinearRegression);
      //@ts-expect-error
      expect(MLRConfigured.settings.test).toEqual('prop');
    });
  });
  describe('predict',()=>{
    it('should make predictions from trained model', async () => {
      const predictions = await trainedMLR.predict(input_x);
      const shape = trainedMLR.getInputShape(predictions);
      // console.log('nnLR.layers', nnLR.layers);
      // console.log({
      //   predictions,
      //   shape,
      // });
      expect(predictions).toHaveLength(input_x.length);
      // expect(trainedMLR.layers).toHaveLength(1);
      const descaledPredictions = predictions.map(DataSet.scalers.get('price').descale);
      // console.log({
      //   descaledPredictions,
      //   shape,
      //   explain: trainedMLR.explain()
      // },trainedMLR.model.getParams())
      //@ts-expect-error
      expect(descaledPredictions[ 0 ]).toBeWithinRange(600000, 670000);
      //@ts-expect-error
      expect(descaledPredictions[ 1 ]).toBeWithinRange(160000, 220000);
      return true;
    });
  })
});