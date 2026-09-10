import {test} from 'node:test';
import assert from 'node:assert/strict';
import {layoutNodes} from '../../pss/web/static/graph.js';
import {examples,visibleNodes} from '../../pss/web/static/examples.js';

test('every authored thought can become the central selected thought',()=>{
  for(const board of examples)for(const selected of board.nodes){
    const positions=layoutNodes(board.nodes,selected.id);
    assert.equal(positions.size,board.nodes.length);
    assert.equal(positions.get(selected.id).kind,'focus');
    assert.equal(positions.get(selected.id).x,588);
    assert.equal([...positions.values()].filter(p=>p.kind==='focus').length,1);
    const values=[...positions.values()];
    for(let i=0;i<values.length;i++)for(let j=i+1;j<values.length;j++){
      const a=values[i],b=values[j];
      assert.ok(a.x+a.w<=b.x||b.x+b.w<=a.x||a.y+a.h<=b.y||b.y+b.h<=a.y,'cards must not overlap');
    }
  }
});
test('pruning only lays out the surviving graph and missing selection falls back safely',()=>{
  const nodes=visibleNodes(examples[1].nodes,new Set(['gift']));
  const positions=layoutNodes(nodes,'gift');
  assert.equal(positions.size,5);assert.ok(!positions.has('gift'));assert.ok(!positions.has('map'));
  assert.equal(positions.get('root').kind,'focus');
  assert.equal(layoutNodes([],null).size,0);
});
test('the first-contact target retains complete authored stories and provenance',()=>{
  const board=examples[1];assert.equal(board.prompt,'What if first contact arrives as a joke?');
  assert.equal(board.nodes.find(n=>n.id==='gift').title,'A language made of punchlines');
  for(const n of board.nodes){assert.ok(n.body.includes(n.preview));if(n.next)assert.equal(board.nodes.find(c=>c.id===n.next).parent,n.id);}
});
