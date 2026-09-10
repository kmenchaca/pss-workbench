export function element(tag, className, text) {
  const el = document.createElement(tag); if(className) el.className = className; if(text !== undefined) el.textContent = text; return el;
}
// Positions express real graph relationships, not simulated activity.
export function layoutNodes(nodes, selected) {
  const focus=nodes.find(n=>n.id===selected)||nodes[0];if(!focus)return new Map();
  const root=nodes.find(n=>!n.parent),siblings=nodes.filter(n=>n.id!==focus.id&&n.parent===focus.parent);
  const positions=new Map([[focus.id,{x:588,y:190,w:340,h:370,kind:'focus'}]]);
  const sides=[{x:265,y:275,w:280,h:270,kind:'sibling'},{x:988,y:275,w:280,h:270,kind:'sibling'}];
  siblings.slice(0,2).forEach((n,i)=>positions.set(n.id,sides[i]));
  if(root&&root.id!==focus.id)positions.set(root.id,{x:642,y:32,w:225,h:115,kind:'small'});
  const peripheral=[{x:50,y:180},{x:1293,y:315},{x:1293,y:550},{x:50,y:550},{x:350,y:650},{x:820,y:650}];
  nodes.filter(n=>!positions.has(n.id)).forEach((n,i)=>positions.set(n.id,{...(peripheral[i]||{x:50+(i%6)*230,y:750+Math.floor(i/6)*170}),w:195,h:130,kind:'small'}));
  return positions;
}
export function renderGraph(container,nodes,selected,compared,onSelect,onCompare) {
  const focus=document.activeElement?.dataset?.node,compareFocus=document.activeElement?.dataset?.compare;
  container.replaceChildren();const positions=layoutNodes(nodes,selected);
  const height=Math.max(735,...[...positions.values()].map(p=>p.y+p.h+24));container.style.height=`${height}px`;
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('graph-lines');svg.setAttribute('viewBox',`0 0 1488 ${height}`);svg.setAttribute('aria-hidden','true');container.append(svg);
  const cards=new Map();
  nodes.forEach((n,i)=>{
    const p=positions.get(n.id),card=element('article',`node ${p.kind} ${n.parent?'':'root-node'} ${n.id===selected?'is-selected':''}`);Object.assign(card.style,{left:`${p.x}px`,top:`${p.y}px`,width:`${p.w}px`,minHeight:`${p.h}px`});
    if(p.kind!=='small'){const badge=element('span','branch-badge');const icon=element('img');icon.src=`/static/assets/icons/${n.id==='repair'?'mic-vocal':n.id==='silence'?'ticket':'smile'}.svg`;icon.alt='';icon.width=30;icon.height=30;badge.append(icon);card.append(badge);}
    const b=element('button','node-select');b.dataset.node=n.id;b.setAttribute('aria-label',`${n.title}. Select branch`);b.setAttribute('aria-pressed',String(n.id===selected));b.addEventListener('click',()=>onSelect(n.id));
    b.append(element('span','node-lens',n.id===selected?'SELECTED BRANCH':n.lens||n.status||'BRANCH'),element('strong','node-title',n.title),element('span','node-summary',p.kind==='small'?n.summary:n.preview||n.summary));
    if(p.kind!=='small'&&n.tradeoff)b.append(element('span','node-tradeoff',n.angle||n.tradeoff));
    const foot=element('div','node-foot');foot.append(element('span','node-number',`${String(i+1).padStart(2,'0')} · ${n.parent?'BRANCH':'ROOT'}`));
    if(n.parent){const label=element('label','compare-check'),box=document.createElement('input');box.type='checkbox';box.dataset.compare=n.id;box.checked=compared.has(n.id);box.setAttribute('aria-label',`Compare ${n.title}`);box.addEventListener('change',()=>onCompare(n.id));label.append(box,document.createTextNode('Compare'));foot.append(label);}
    card.append(b,foot);container.append(card);cards.set(n.id,card);
  });
  // Measure after text wrapping. Preserve primary-card positions and move small
  // cards down until their full rendered bounds clear every earlier card.
  if(!window.matchMedia('(max-width: 650px)').matches){
    const placed=[];
    const graphBox=container.getBoundingClientRect(),questionBox=document.querySelector('.question')?.getBoundingClientRect();
    const scale=graphBox.width/container.offsetWidth||1;
    const question=questionBox?{x:(questionBox.left-graphBox.left)/scale,y:(questionBox.top-graphBox.top)/scale,w:questionBox.width/scale,h:questionBox.height/scale}:null;
    const ordered=[...nodes].sort((a,b)=>(positions.get(a.id).kind==='small')-(positions.get(b.id).kind==='small'));
    for(const n of ordered){const p=positions.get(n.id);p.h=cards.get(n.id).offsetHeight;p.w=cards.get(n.id).offsetWidth;let collision;
      const obstacles=question&&p.kind==='small'?[question,...placed]:placed;
      while((collision=obstacles.find(q=>p.x<q.x+q.w+20&&p.x+p.w+20>q.x&&p.y<q.y+q.h+20&&p.y+p.h+20>q.y)))p.y=collision.y+collision.h+24;
      cards.get(n.id).style.top=`${p.y}px`;placed.push(p);
    }
    const measuredHeight=Math.max(735,...placed.map(p=>p.y+p.h+24));container.style.height=`${measuredHeight}px`;svg.setAttribute('viewBox',`0 0 1488 ${measuredHeight}`);
    nodes.forEach(n=>{const p=positions.get(n.id),parent=positions.get(n.parent);if(!parent)return;
      const path=document.createElementNS(svg.namespaceURI,'path');let sx,sy,ex,ey,c1x,c1y,c2x,c2y;
      if(p.x>=parent.x+parent.w||p.x+p.w<=parent.x){const right=p.x>parent.x;sx=parent.x+(right?parent.w:0);sy=parent.y+parent.h*.65;ex=p.x+(right?0:p.w);ey=p.y+p.h*.5;const bend=Math.min(100,Math.abs(ex-sx)*.5);c1x=sx+(right?bend:-bend);c2x=ex+(right?-bend:bend);c1y=sy;c2y=ey;}
      else{const below=p.y>parent.y;sx=parent.x+parent.w*.5;sy=parent.y+(below?parent.h:0);ex=p.x+p.w*.5;ey=p.y+(below?0:p.h);c1x=sx;c2x=ex;c1y=sy+(below?45:-45);c2y=ey+(below?-45:45);}
      path.setAttribute('d',`M ${sx} ${sy} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${ex} ${ey}`);path.classList.toggle('selected-edge',n.id===selected||n.parent===selected);svg.append(path);
    });
  }
  if(focus)[...container.querySelectorAll('[data-node]')].find(el=>el.dataset.node===focus)?.focus({preventScroll:true});
  if(compareFocus)[...container.querySelectorAll('[data-compare]')].find(el=>el.dataset.compare===compareFocus)?.focus({preventScroll:true});
}
