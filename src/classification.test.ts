import path from 'path';
import * as ms from '@modelx/data';
import { DeepLearningClassification, } from './index';
import '@tensorflow/tfjs-node';
import '@tensorflow/tfjs-backend-wasm';
const independentVariables = [
  'sepal_length_cm',
  'sepal_width_cm',
  'petal_length_cm',
  'petal_width_cm',
];
const dependentVariables = [
  'plant_Iris-setosa',
  'plant_Iris-versicolor',
  'plant_Iris-virginica',
];
const columns = independentVariables.concat(dependentVariables);
let housingDataCSV;
let DataSet;
let x_matrix;
let y_matrix;
let nnClassification;
let nnClassificationModel;
const fit = {
  epochs: 100,
  batchSize: 5,
};
const encodedAnswers = {
  'Iris-setosa': [1, 0, 0, ],
  'Iris-versicolor': [0, 1, 0, ],
  'Iris-virginica': [0, 0, 1, ],
};
const input_x = [
  [5.1, 3.5, 1.4, 0.2, ],
  [6.3,3.3,6.0,2.5, ],
  [5.6, 3.0, 4.5, 1.5, ],
  [5.0, 3.2, 1.2, 0.2, ],
  [4.5, 2.3, 1.3, 0.3, ],
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
/** @test {DeepLearningClassification} */
describe('DeepLearningClassification', function () {
  beforeAll(async function () {
    /**
     * encodedData = [ 
     *  { sepal_length_cm: 5.1,
         sepal_width_cm: 3.5,
        petal_length_cm: 1.4,
        petal_width_cm: 0.2,
        plant: 'Iris-setosa',
        'plant_Iris-setosa': 1,
        'plant_Iris-versicolor': 0,
        'plant_Iris-virginica': 0 },
        ...
        { sepal_length_cm: 5.9,
        sepal_width_cm: 3,
        petal_length_cm: 4.2,
        petal_width_cm: 1.5,
        plant: 'Iris-versicolor',
        'plant_Iris-setosa': 0,
        'plant_Iris-versicolor': 1,
        'plant_Iris-virginica': 0 },
      ];
    */
    housingDataCSV = await ms.csv.loadCSV(path.join(__dirname,'/test/mock/data/iris_data.csv'));
    DataSet = new ms.DataSet(housingDataCSV);
    // DataSet.fitColumns({
    //   columns: columns.map(scaleColumnMap),
    //   returnData:false,
    // });
    const encodedData = DataSet.fitColumns({
      columns: [
        {
          name: 'plant',
          options: {
            strategy: 'onehot',
          },
        },
      ],
      returnData:true,
    });
    x_matrix = DataSet.columnMatrix(independentVariables); 
    y_matrix = DataSet.columnMatrix(dependentVariables);
    /*
    x_matrix = [
      [ 5.1, 3.5, 1.4, 0.2 ],
      [ 4.9, 3, 1.4, 0.2 ],
      [ 4.7, 3.2, 1.3, 0.2 ],
      ...
    ]; 
    y_matrix = [
      [ 1, 0, 0 ],
      [ 1, 0, 0 ],
      [ 1, 0, 0 ],
      ...
    ] 
    */
    // console.log({ x_matrix, y_matrix, });

    nnClassification = new DeepLearningClassification({ fit, });
    await nnClassification.tf.setBackend('wasm');
    console.log('nnClassification.tf.getBackend()',nnClassification.tf.getBackend());


    nnClassificationModel = await nnClassification.train(x_matrix, y_matrix);
  },120000);
  /** @test {DeepLearningClassification#constructor} */
  describe('constructor', () => {
    it('should export a named module class', async () => {
      const NN = new DeepLearningClassification();
      await NN.tf.setBackend('wasm');
      console.log('NN.tf.getBackend()',NN.tf.getBackend());

      //@ts-ignore
      const NNConfigured = new DeepLearningClassification({ test: 'prop', });
      expect(typeof DeepLearningClassification).toBe('function');
      expect(NN).toBeInstanceOf(DeepLearningClassification);
      expect(NNConfigured.settings.test).toBe('prop');
    });
  });
  /** @test {DeepLearningClassification#generateLayers} */
  describe('generateLayers', () => {
    it('should generate a classification network', async () => {
      const predictions = await nnClassification.predict(input_x);
      const answers = await nnClassification.predict(input_x, {
        probability:false,
      });
      const shape = nnClassification.getInputShape(predictions);
      // console.log('nnClassification.layers', nnClassification.layers);
      // console.log({
      //   predictions_unscaled,
      //   predictions,
      //   shape,
      // });
      
      // const probabilities = ms.DataSet.reverseColumnMatrix({
      //   vectors: predictions,
      //   labels: dependentVariables,
      // });
      // const results = ms.DataSet.reverseColumnMatrix({
      //   vectors: answers,
      //   labels: dependentVariables,
      // });
      // console.log({
      //   predictions,
      //   // probabilities,
      //   answers,
      //   // results,
      //   shape,
      // });
      expect(predictions).toHaveLength(input_x.length);
      expect(nnClassification.layers).toHaveLength(2);
      expect(shape).toMatchObject([5, 3,]);
      expect(answers[ 0 ]).toMatchObject(encodedAnswers[ 'Iris-setosa' ]);
      // expect(answers[ 1 ]).to.eql(encodedAnswers[ 'Iris-virginica' ]);
      // expect(answers[ 2 ]).to.eql(encodedAnswers[ 'Iris-versicolor' ]);
      // expect(answers[ 3 ]).to.eql(encodedAnswers[ 'Iris-setosa' ]);
      // expect(answers[ 4 ]).to.eql(encodedAnswers[ 'Iris-setosa' ]);
      return true;
    });
    it('should generate a network from layers', async () => { 
      const nnClassificationCustom = new DeepLearningClassification({ layerPreference:'custom', fit, });
      await nnClassificationCustom.tf.setBackend('cpu');
      console.log('nnClassificationCustom.tf.getBackend()',nnClassificationCustom.tf.getBackend());

      await nnClassificationCustom.train(x_matrix, y_matrix, nnClassification.layers);
      expect(nnClassificationCustom.layers).toHaveLength(2);
    },120000);
  });
});