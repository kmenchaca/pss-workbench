import test from 'node:test';
import assert from 'node:assert/strict';
import {isCheckpoint,thoughtContent} from '../../pss/web/static/thought-content.js';
test('checkpoint protocol is separated from the readable thought, not deleted from conversation',()=>{
  const messages=[{role:'assistant',content:'**Living Archive**\n\nVisitors shape a shared exhibit.'},{role:'assistant',content:'{"action":"continue"}'}];
  const result=thoughtContent({messages});
  assert.equal(result.title,'Living Archive');assert.equal(result.summary,'Visitors shape a shared exhibit.');assert.ok(!result.body.includes('continue'));assert.equal(messages.length,2);assert.match(result.bodySource,/checkpoints/);
});
test('final model output is preserved verbatim',()=>{const output='**An approach**\n\nA full, precise final result.';assert.equal(thoughtContent({output,messages:[]}).body,output);});
test('normal JSON answers are never mistaken for engine protocol',()=>{assert.equal(isCheckpoint('{"action":"build","plan":"A"}'),false);assert.equal(isCheckpoint('{"action":"continue","unrelated":"user data"}'),false);assert.equal(isCheckpoint('ordinary prose'),false);});
