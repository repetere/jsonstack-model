//@ts-nocheck
import { FeatureEmbedding, } from './index';
import * as tf from '@tensorflow/tfjs';

export function toBeWithinRange(received, floor, ceiling) {
  const pass = received >= floor && received <= ceiling;
  if (pass) {
    return {
      message: () =>
        `expected ${received} not to be within range ${floor} - ${ceiling}`,
      pass: true,
    };
  } else {
    return {
      message: () =>
        `expected ${received} to be within range ${floor} - ${ceiling}`,
      pass: false,
    };
  }
};

expect.extend({
  toBeWithinRange,
});

describe('toBeWithinRage', () => {
  it('numeric ranges', () => {
    //@ts-expect-error
    expect(100).toBeWithinRange(90, 110);
    //@ts-expect-error
    expect(101).not.toBeWithinRange(0, 100);
    expect({ apples: 6, bananas: 3 }).toEqual({
      //@ts-expect-error
      apples: expect.toBeWithinRange(1, 10),
      //@ts-expect-error
      bananas: expect.not.toBeWithinRange(11, 20),
    });
  });
  // it('should add to existing data', async () => {
  //   const addToExisting = await FeatureEmbedding.getFeatureDataSet({
  //     inputMatrixFeatures: [['new1', 'new2', 'old1', 'old2']],
  //     initialIdToFeature: { 1: 'old1', 2: 'old2' },
  //     initialFeatureToId: { old1: 1, old2: 2 },
  //   });
  //   console.log('addToExisting', addToExisting);
  //   expect(addToExisting.numberOfFeatures).toBe(5);
  //   expect(addToExisting.featureIds).toMatchObject([[3, 4, 1, 2]]);
  //   expect(addToExisting.featureToId).toMatchObject({ PAD: 0, old1: 1, old2: 2, new1: 3, new2: 4 });
  //   expect(addToExisting.idToFeature).toMatchObject({ '0': 'PAD', '1': 'old1', '2': 'old2', '3': 'new1', '4': 'new2' });
  // });
});