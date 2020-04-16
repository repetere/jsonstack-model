import path from 'path';
import * as ms from '@modelx/data';
import { MultipleLinearRegression, } from './index';
const independentVariables = ['sqft', 'bedrooms', ];
const dependentVariables = ['price',];
let housingDataCSV;
let input_x;
let DataSet;
let x_matrix;
let y_matrix;
let trainedMLR;
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
/** @test {MultipleLinearRegression} */
describe('MultipleLinearRegression', function () {
  beforeAll(async function () {
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
    trainedMLR = new MultipleLinearRegression({
      fit: {
        epochs: 100,
        batchSize: 5,
      },
    });
    trainedMLRModel = await trainedMLR.train(x_matrix, y_matrix);
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
  /** @test {MultipleLinearRegression#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const MLR = new MultipleLinearRegression({
        fit: {
          epochs: 200,
          batchSize: 5,
        },
      });
      const MLRConfigured = new MultipleLinearRegression({ test: 'prop', }, {});
      expect(typeof MultipleLinearRegression).toBe('function');
      expect(MLR).toBeInstanceOf(MultipleLinearRegression);
      expect(MLRConfigured.settings.test).toEqual('prop');
    });
  });
  /** @test {MultipleLinearRegression#generateLayers} */
  describe('generateLayers', () => {
    it('should generate a classification network', async () => {
      const predictions = await trainedMLR.predict(input_x);
      const shape = trainedMLR.getInputShape(predictions);
      // console.log('nnLR.layers', nnLR.layers);
      // console.log({
      //   predictions,
      //   shape,
      // });
      expect(predictions).toHaveLength(input_x.length);
      expect(trainedMLR.layers).toHaveLength(1);
      const descaledPredictions = predictions.map(DataSet.scalers.get('price').descale);
      expect(descaledPredictions[ 0 ]).toBeCloseTo(630000, 20000);
      expect(descaledPredictions[ 1 ]).toBeCloseTo(190000, 10000);
      return true;
    });
    it('should generate a network from layers', async () => {
      const nnLRCustom = new MultipleLinearRegression({
        type: 'custom',
        fit: {
          epochs: 10,
          batchSize: 5,
        },
      });
      await nnLRCustom.train(x_matrix, y_matrix, trainedMLR.layers);
      expect(nnLRCustom.layers).toHaveLength(1);
      return true;
    }, 20000);
  });
});