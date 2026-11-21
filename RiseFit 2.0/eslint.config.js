// https://docs.expo.dev/guides/using-eslint/
const { defineConfig } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

module.exports = defineConfig([
  expoConfig,
  {
    ignores: ['dist/*'],
  },
  {
    rules: {
      // React Compiler rules that conflict with imperative animation APIs we use
      // (Reanimated shared values, RN Animated) and with Clock reads in memoised
      // selectors. Component state still follows the normal hooks rules.
      'react-hooks/immutability': 'off',
      'react-hooks/purity': 'off',
    },
  },
]);
