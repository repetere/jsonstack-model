//@ts-nocheck
import path from 'path';
import * as ms from '@jsonstack/data';
import * as tf from '@tensorflow/tfjs-node';
import * as scikit from 'scikitjs/dist/node';
import { MachineLearningLogisticRegression, setBackend, setScikit, } from './index';
import { toBeWithinRange, } from './jest.test';
expect.extend({ toBeWithinRange });
setBackend(tf);
scikit.setBackend(tf);
setScikit(scikit);

// import '@tensorflow/tfjs-node-gpu';
// import '@tensorflow/tfjs-backend-cpu';
// import '@tensorflow/tfjs-backend-wasm';
const independentVariables = [
  'Age',
  'EstimatedSalary',
];
const dependentVariables = [
  'Purchased',
];
let CSVData;
let DataSet;
let x_matrix;
let y_matrix;
let nnLR;
let nnLRClass;
let nnLRReg;
let nnLRModel;
let nnLRClassModel;
let nnLRRegModel;
const encodedAnswers = {
  'yes': [1,],
  'no': [0,],
};
const fit = {
  epochs: 10,
  batchSize: 5,
};
const input_x = [
  [-0.062482849427819266, 0.30083326827486173, ], //0
  [0.7960601198093905, -1.1069168538010206, ], //1
  [0.7960601198093905, 0.12486450301537644, ], //0
  [0.4144854668150751, -0.49102617539282206, ], //0
  [0.3190918035664962, 0.5061301610775946, ], //1
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
/** @test {MachineLearningLogisticRegression} */
describe('MachineLearningLogisticRegression', function () {
  beforeAll(async function () {
    const fpath = `${path.join(__dirname, '/test/mock/data/social_network_ads.csv')}`;
    CSVData = await ms.csv.loadCSV(fpath);
    DataSet = new ms.DataSet(CSVData);
    const scaledData = DataSet.fitColumns({
      columns: independentVariables.map(scaleColumnMap),
      returnData:true,
    });
    /*
    scaledData = [ 
      { 'User ID': 15624510,
         Gender: 'Male',
         Age: -1.7795687879022388,
         EstimatedSalary: -1.4881825118632386,
         Purchased: 0 },
      { 'User ID': 15810944,
         Gender: 'Male',
         Age: -0.253270175924977,
         EstimatedSalary: -1.458854384319991,
         Purchased: 0 },
      ...
    ];
    */
    x_matrix = DataSet.columnMatrix(independentVariables); 
    y_matrix = DataSet.columnMatrix(dependentVariables);
    /*
    x_matrix = [
      [ -1.7795687879022388, -1.4881825118632386 ],
      [ -0.253270175924977, -1.458854384319991 ],
      ...
    ]; 
    y_matrix = [
      [ 0 ],
      [ 0 ],
      ...
    ] 
    */
    // console.log({ x_matrix, y_matrix, });

    nnLR = new MachineLearningLogisticRegression({ fit,  });
    nnLRClass = new MachineLearningLogisticRegression({ type: 'class', fit, });
    nnLRReg = new MachineLearningLogisticRegression({ type: 'l1l2', fit, });
    // await nnLR.tf.setBackend('cpu')
    // await nnLRClass.tf.setBackend('cpu')
    // await nnLRReg.tf.setBackend('cpu')
    // await nnLR.tf.setBackend('wasm')
    // await nnLRClass.tf.setBackend('wasm')
    // console.log('nnLR.tf.getBackend()',nnLR.tf.getBackend())
    // console.log('nnLRClass.tf.getBackend()',nnLRClass.tf.getBackend())
    // console.log('nnLRReg.tf.getBackend()',nnLRReg.tf.getBackend())
    const models = await Promise.all([
      nnLR.train(x_matrix, y_matrix),
      nnLRClass.train(x_matrix, y_matrix),
      nnLRReg.train(x_matrix, y_matrix),
    ]);
    nnLRModel = models[ 0 ];
    nnLRClassModel = models[ 1 ];
    nnLRRegModel = models[ 2 ];
  },120000);
  /** @test {MachineLearningLogisticRegression#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const NN = new MachineLearningLogisticRegression();
      //@ts-expect-error
      const NNConfigured = new MachineLearningLogisticRegression({ test: 'prop', });
      expect(typeof MachineLearningLogisticRegression).toBe('function');
      expect(NN).toBeInstanceOf(MachineLearningLogisticRegression);
      //@ts-expect-error
      expect(NNConfigured.settings.test).toEqual('prop');
    });
  });
  /** @test {MachineLearningLogisticRegression#generateLayers} */
  describe('generateLayers', () => {
    it('should generate a classification network', async () => {
      //TODO: @dcrmls you could the shape here
      /*
      
      const predictions = await nnLR.predict(input_x);
      const answers = await nnLR.predict(input_x, {
        probability:false,
      });
      const shape = nnLR.getInputShape(predictions);
      
      expect(predictions).toHaveLength(input_x.length);
      expect(nnLR.layers).toHaveLength(1);
      expect(shape).toEqual([5, 1, ]);
      */
      // expect(answers[ 0 ]).to.eql(encodedAnswers[ 'Iris-setosa' ]);
      return true;
    });

  });
});