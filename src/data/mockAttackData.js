// mockAttackData.js
// 실제 백엔드 연결 전 사용하는 mock 데이터

export const SAMPLES = {
  'sample-1': { trueLabel: 7, description: 'Digit 7' },
  'sample-2': { trueLabel: 3, description: 'Digit 3' },
  'sample-3': { trueLabel: 5, description: 'Digit 5' },
}

export const ORIGINAL_PROBS = {
  'sample-1': { '0': 0.002, '1': 0.006, '2': 0.001, '3': 0.003, '4': 0.001, '5': 0.002, '6': 0.001, '7': 0.984, '8': 0.000, '9': 0.000 },
  'sample-2': { '0': 0.001, '1': 0.002, '2': 0.008, '3': 0.971, '4': 0.003, '5': 0.007, '6': 0.002, '7': 0.003, '8': 0.002, '9': 0.001 },
  'sample-3': { '0': 0.002, '1': 0.001, '2': 0.004, '3': 0.005, '4': 0.002, '5': 0.969, '6': 0.008, '7': 0.003, '8': 0.004, '9': 0.002 },
}

export const ADVERSARIAL_PROBS = {
  'sample-1': { '0': 0.004, '1': 0.912, '2': 0.008, '3': 0.021, '4': 0.003, '5': 0.010, '6': 0.002, '7': 0.038, '8': 0.001, '9': 0.001 },
  'sample-2': { '0': 0.003, '1': 0.006, '2': 0.891, '3': 0.042, '4': 0.018, '5': 0.012, '6': 0.009, '7': 0.005, '8': 0.010, '9': 0.004 },
  'sample-3': { '0': 0.021, '1': 0.004, '2': 0.007, '3': 0.009, '4': 0.003, '5': 0.028, '6': 0.892, '7': 0.011, '8': 0.018, '9': 0.007 },
}

export const ADVERSARIAL_PREDS = {
  'sample-1': 1,
  'sample-2': 2,
  'sample-3': 6,
}

export const MOCK_EXPERIMENT_LOG = [
  {
    id: 'exp-001',
    time: '10:23:11',
    sample: 'sample-1',
    origPred: 7,
    advPred: 1,
    epsilon: 0.10,
    confBefore: 0.984,
    confAfter: 0.912,
    success: true,
  },
  {
    id: 'exp-002',
    time: '10:25:44',
    sample: 'sample-2',
    origPred: 3,
    advPred: 2,
    epsilon: 0.15,
    confBefore: 0.971,
    confAfter: 0.891,
    success: true,
  },
  {
    id: 'exp-003',
    time: '10:28:02',
    sample: 'sample-3',
    origPred: 5,
    advPred: 6,
    epsilon: 0.08,
    confBefore: 0.969,
    confAfter: 0.892,
    success: true,
  },
  {
    id: 'exp-004',
    time: '10:31:17',
    sample: 'sample-1',
    origPred: 7,
    advPred: 7,
    epsilon: 0.02,
    confBefore: 0.984,
    confAfter: 0.901,
    success: false,
  },
  {
    id: 'exp-005',
    time: '10:34:58',
    sample: 'sample-3',
    origPred: 5,
    advPred: 6,
    epsilon: 0.25,
    confBefore: 0.969,
    confAfter: 0.955,
    success: true,
  },
]
