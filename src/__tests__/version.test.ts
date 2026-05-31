import { describe, expect, it } from 'vitest';
import packageJson from '../../package.json';
import { VERSION } from '../index';

describe('VERSION', () => {
  it('matches package.json', () => {
    expect(VERSION).toBe(packageJson.version);
  });
});
