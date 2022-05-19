import { getBackend, setBackend } from "./tensorflow_singleton";

describe('tensorflow singleton',()=>{
  describe('getBackend',()=>{
    it('should throw an error if tensorflow has not been set',()=>{
      expect(getBackend.bind(null)).toThrowError(/Looks like you are/);
    });
  });
  describe('setBackend',()=>{
    const tensorflowMockBackend = {};
    it('should set a tf singleton',()=>{
      setBackend(tensorflowMockBackend);
      expect(getBackend()).toBe(tensorflowMockBackend);
    });
  });
});