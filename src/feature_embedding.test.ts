import path from 'path';
import * as ms from '@modelx/data';
import { FeatureEmbedding, } from './index';
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs-node';
import { stop_words, } from './test/mock/data/stopwords';
// const independentVariables = [
//   'sepal_length_cm',
//   'sepal_width_cm',
//   'petal_length_cm',
//   'petal_width_cm',
// ];
// const dependentVariables = [
//   'plant_Iris-setosa',
//   'plant_Iris-versicolor',
//   'plant_Iris-virginica',
// ];
// const columns = independentVariables.concat(dependentVariables);
// let housingDataCSV;
// let DataSet;
// let x_matrix;
// let y_matrix;
// let nnClassification;
// let nnClassificationModel;
// const fit = {
//   epochs: 100,
//   batchSize: 5,
// };
// const encodedAnswers = {
//   'Iris-setosa': [1, 0, 0, ],
//   'Iris-versicolor': [0, 1, 0, ],
//   'Iris-virginica': [0, 0, 1, ],
// };
// const input_x = [
//   [5.1, 3.5, 1.4, 0.2, ],
//   [6.3,3.3,6.0,2.5, ],
//   [5.6, 3.0, 4.5, 1.5, ],
//   [5.0, 3.2, 1.2, 0.2, ],
//   [4.5, 2.3, 1.3, 0.3, ],
// ];
// function scaleColumnMap(columnName) {
//   return {
//     name: columnName,
//     options: {
//       strategy: 'scale',
//       scaleOptions: {
//         strategy:'standard',
//       },
//     },
//   };
// }

const norm_bible = [
  'food used hunger',
  'food used eat',
  'need food eat',
  'need food hunger',
  'want food hunger',
  'want food eat',
  'want car drive',
  'need car drive',
  'need car travel',
  'want car travel',
  'use fly travel fast',
  'fast fly travel',
  'need food play',
  'need food live',
  'need eat live',
  'tesla kind car',
  'tesla fast car',
  'tesla is an electric car',
  'electric is fast',
  'car fly walk ways travel',
  'want tesla drive car',
  'want food solve hunger',
  'need eat food hunger',
  'want tesla drive fast far',
  'need plane fly'];
const norm_bible_matrix = norm_bible.map(nb => nb.split(' ').filter(nb => !stop_words.includes(nb)));
// console.log({ norm_bible_matrix });
/** @test {FeatureEmbedding} */
describe('FeatureEmbedding', function () {
  beforeAll(async function () {
    const conf = await FeatureEmbedding.getFeatureDataSet({ inputMatrixFeatures: norm_bible_matrix, });
    const {
      word2id,
      id2word,
      wids,
      vocab_size,
    } = conf;
    // console.log({ conf });
    // console.log('wids',wids);
    
    const cxt = await FeatureEmbedding.getContextPairs({ tf, vocab_size, inputMatrix: wids });
    // cxt.x.forEach((xitem, i) => {
    //   console.log(xitem, cxt.y[i]);
    // })
    // console.log('cxt.x.length',cxt.x.length);
    // console.log('cxt.y.length',cxt.y.length);
    // console.log('cxt',cxt);
    // /**
    //  * encodedData = [ 
    //  *  { sepal_length_cm: 5.1,
    //      sepal_width_cm: 3.5,
    //     petal_length_cm: 1.4,
    //     petal_width_cm: 0.2,
    //     plant: 'Iris-setosa',
    //     'plant_Iris-setosa': 1,
    //     'plant_Iris-versicolor': 0,
    //     'plant_Iris-virginica': 0 },
    //     ...
    //     { sepal_length_cm: 5.9,
    //     sepal_width_cm: 3,
    //     petal_length_cm: 4.2,
    //     petal_width_cm: 1.5,
    //     plant: 'Iris-versicolor',
    //     'plant_Iris-setosa': 0,
    //     'plant_Iris-versicolor': 1,
    //     'plant_Iris-virginica': 0 },
    //   ];
    // */
    // housingDataCSV = await ms.csv.loadCSV(path.join(__dirname,'/test/mock/data/iris_data.csv'));
    // DataSet = new ms.DataSet(housingDataCSV);
    // // DataSet.fitColumns({
    // //   columns: columns.map(scaleColumnMap),
    // //   returnData:false,
    // // });
    // const encodedData = DataSet.fitColumns({
    //   columns: [
    //     {
    //       name: 'plant',
    //       options: {
    //         strategy: 'onehot',
    //       },
    //     },
    //   ],
    //   returnData:true,
    // });
    // x_matrix = DataSet.columnMatrix(independentVariables); 
    // y_matrix = DataSet.columnMatrix(dependentVariables);
    // /*
    // x_matrix = [
    //   [ 5.1, 3.5, 1.4, 0.2 ],
    //   [ 4.9, 3, 1.4, 0.2 ],
    //   [ 4.7, 3.2, 1.3, 0.2 ],
    //   ...
    // ]; 
    // y_matrix = [
    //   [ 1, 0, 0 ],
    //   [ 1, 0, 0 ],
    //   [ 1, 0, 0 ],
    //   ...
    // ] 
    // */
    // // console.log({ x_matrix, y_matrix, });

    // nnClassification = new FeatureEmbedding({ fit, });
    // nnClassificationModel = await nnClassification.train(x_matrix, y_matrix);
  }, 120000);
  /** @test {FeatureEmbedding#getMergedArray} */
  describe('static getMergedArray', () => {
    it('should merge two arrays', () => {
      const b = [0, 0, 0, 0];
      const m = [1, 2];
      const b1 = [0, 0, 0, 0];
      const m1 = [];
      const b2 = [0, 0];
      const m2 = [5, 6, 7, 8];
      const merged = FeatureEmbedding.getMergedArray(b, m);
      const merged1 = FeatureEmbedding.getMergedArray(b1, m1);
      const merged2 = FeatureEmbedding.getMergedArray(b2, m2);
      expect(merged).toMatchObject([1, 2, 0, 0]);
      expect(merged1).toMatchObject([0, 0, 0, 0]);
      expect(merged2).toMatchObject([5, 6, 7, 8]);
    });
    it('should append and merge two arrays', () => {
      const b = [0, 0, 0, 0];
      const m = [1, 2];
      const b1 = [0, 0, 0, 0];
      const m1 = [];
      const b2 = [0, 0];
      const m2 = [5, 6, 7, 8];
      const merged = FeatureEmbedding.getMergedArray(b, m, true);
      const merged1 = FeatureEmbedding.getMergedArray(b1, m1, true);
      const merged2 = FeatureEmbedding.getMergedArray(b2, m2, true);
      // console.log({merged,merged1,merged2})
      expect(merged).toMatchObject([0, 0, 1, 2,]);
      expect(merged1).toMatchObject([0, 0, 0, 0]);
      expect(merged2).toMatchObject([5, 6, 7, 8]);
    });
    it('should handle empty arrays', () => {
      const merged = FeatureEmbedding.getMergedArray();
      expect(merged).toMatchObject([]);
    });
  });
  /** @test {FeatureEmbedding#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const NN = new FeatureEmbedding();
      //@ts-ignore
      const NNConfigured = new FeatureEmbedding({ test: 'prop', });
      expect(typeof FeatureEmbedding).toBe('function');
      expect(NN).toBeInstanceOf(FeatureEmbedding);
      expect(NNConfigured.settings.test).toBe('prop');
    });
  });
  describe('generate feature embedder', () => {
    it('should train a model', async () => {
      const FE = new FeatureEmbedding({
        fit: {
          epochs: 15,
          batchSize: 1,
          callbacks: {
            // onTrainBegin: (logs:any) => console.log('onTrainBegin',{logs, }),
            // // called when training ends.
            onTrainEnd: (logs:any) => console.log('onTrainEnd',{logs, }),
            // //called at the start of every epoch.
            onEpochBegin: (epoch: number, logs: any) => console.log('onEpochBegin',{logs, epoch, }),
            // //called at the end of every epoch.
            // onEpochEnd: (epoch: number, logs: any) => console.log('onEpochEnd',{logs, epoch, }),
            // //called at the start of every batch.
            // onBatchBegin:(batch:number, logs: any)=> console.log('onBatchBegin',{logs,  batch}),
            // //called at the end of every batch.
            // onBatchEnd:(batch:number, logs: any)=> console.log('onBatchEnd',{logs,  batch}),
            // //called every yieldEvery milliseconds with the current epoch, batch and logs. The logs are the same as in onBatchEnd(). Note that onYield can skip batches or epochs. See also docs for yieldEvery below.
            // onYield:(epoch:number, batch:number, logs: any)=> console.log('onYield',{logs, epoch, batch}),
          }
        },
      });
      await FE.train(norm_bible_matrix);
      const weights = await FE.predict();
      console.log({ weights });
      const labeled = FE.labelWeights(weights)
      console.log('weights.length', weights.length);
      console.log('FE.id2word', FE.id2word);
      console.log({ labeled });
    })
  })
  /** @test {FeatureEmbedding#generateLayers} */
  describe('generateLayers', () => {
    it('should generate embedding layers', async () => {
      // const predictions = await nnClassification.predict(input_x);
      // const answers = await nnClassification.predict(input_x, {
      //   probability:false,
      // });
      // const shape = nnClassification.getInputShape(predictions);
      // // console.log('nnClassification.layers', nnClassification.layers);
      // // console.log({
      // //   predictions_unscaled,
      // //   predictions,
      // //   shape,
      // // });
      
      // // const probabilities = ms.DataSet.reverseColumnMatrix({
      // //   vectors: predictions,
      // //   labels: dependentVariables,
      // // });
      // // const results = ms.DataSet.reverseColumnMatrix({
      // //   vectors: answers,
      // //   labels: dependentVariables,
      // // });
      // // console.log({
      // //   predictions,
      // //   // probabilities,
      // //   answers,
      // //   // results,
      // //   shape,
      // // });
      // expect(predictions).toHaveLength(input_x.length);
      // expect(nnClassification.layers).toHaveLength(2);
      // expect(shape).toMatchObject([5, 3,]);
      // expect(answers[ 0 ]).toMatchObject(encodedAnswers[ 'Iris-setosa' ]);
      // // expect(answers[ 1 ]).to.eql(encodedAnswers[ 'Iris-virginica' ]);
      // // expect(answers[ 2 ]).to.eql(encodedAnswers[ 'Iris-versicolor' ]);
      // // expect(answers[ 3 ]).to.eql(encodedAnswers[ 'Iris-setosa' ]);
      // // expect(answers[ 4 ]).to.eql(encodedAnswers[ 'Iris-setosa' ]);
      return true;
    });
    // it('should generate a network from layers', async () => { 
    //   const nnClassificationCustom = new FeatureEmbedding({ layerPreference:'custom', fit, });
    //   await nnClassificationCustom.train(x_matrix, y_matrix, nnClassification.layers);
    //   expect(nnClassificationCustom.layers).toHaveLength(2);
    // },120000);
  });
});