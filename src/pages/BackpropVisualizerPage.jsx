import React, { useState, useEffect, useRef } from 'react'

// ─── Design tokens — mirrors project CSS variables exactly ────────────────────
const C = {
  bg:     '#f7f6f3', bg2: '#ffffff', bg3: '#f0eef9',
  border: '#e8e4f0', border2: '#d4cfe8',
  accent: '#7c6ff7', accent2: '#a78bfa',
  green:  '#059669', orange: '#d97706', danger: '#f43f5e', red: '#dc2626',
  text:   '#1a1523', text2: '#6b6580', text3: '#a89fc0',
  tagBg:  '#ede9fe', tagText: '#5b21b6',
}

// ─── Data ─────────────────────────────────────────────────────────────────────
const SAMPLES = [
  { pixels:[1,1,1,1,0,1,1,0,1,1,1,1], target:[1,0], label:'0' },
  { pixels:[0,1,1,1,0,1,1,0,1,1,1,1], target:[1,0], label:'0' },
  { pixels:[1,1,0,1,0,1,1,0,1,1,1,1], target:[1,0], label:'0' },
  { pixels:[1,1,1,1,0,1,1,0,1,1,1,0], target:[1,0], label:'0' },
  { pixels:[1,1,1,1,0,1,1,0,1,0,1,1], target:[1,0], label:'0' },
  { pixels:[0,0,0,0,1,1,0,1,1,0,1,1], target:[0,1], label:'1' },
  { pixels:[0,1,1,1,1,1,0,1,1,0,1,1], target:[0,1], label:'1' },
  { pixels:[0,0,0,1,1,0,1,1,0,1,1,0], target:[0,1], label:'1' },
  { pixels:[0,0,0,1,1,1,1,1,1,1,1,0], target:[0,1], label:'1' },
  { pixels:[0,0,0,1,1,1,1,1,1,0,1,1], target:[0,1], label:'1' },
]

const STEP_META = [
  { label:'INPUT',    icon:'🖼️', desc:'입력 이미지는 12개의 숫자(0 또는 1)로 펼쳐져 신경망에 들어갑니다. 각 픽셀이 하나의 입력 노드가 됩니다.' },
  { label:'FORWARD',  icon:'→',  desc:'각 입력 픽셀에 weight를 곱해 더한 뒤 sigmoid를 적용합니다. 은닉층은 픽셀들의 조합에서 특징을 추출합니다.' },
  { label:'OUTPUT',   icon:'📊', desc:'은닉층 출력이 다시 weight를 거쳐 출력층으로 전달됩니다. out₀는 숫자 0일 가능성, out₁는 숫자 1일 가능성입니다.' },
  { label:'LOSS',     icon:'⚡', desc:'Loss = ½ × Σ(target − output)². Loss가 클수록 예측이 정답과 멀리 떨어져 있습니다.' },
  { label:'BACKPROP', icon:'←',  desc:'오차를 뒤로 전파해 각 weight가 loss에 얼마나 기여했는지 chain rule로 계산합니다. δ₃ → δ₂ 순서로 역전파됩니다.' },
  { label:'UPDATE W', icon:'↺',  desc:'w_new = w_old + η·δ·a_prev. gradient 방향으로 weight를 수정합니다. 이 과정을 반복하면 점점 정답에 가까워집니다.' },
]

const FORMULAS = [
  { label:'입력 펼치기',     expr:'x₁, x₂, ..., x₁₂  ∈ {0, 1}' },
  { label:'은닉층 계산',     expr:'z₂ᵢ = Σ wᵢⱼ · xⱼ + bᵢ\na₂ᵢ = σ(z₂ᵢ) = 1 / (1 + e⁻ᶻ)' },
  { label:'출력층 계산',     expr:'z₃ₖ = Σ wₖᵢ · a₂ᵢ + bₖ\noutₖ = σ(z₃ₖ)' },
  { label:'Loss 계산',       expr:'C = ½ × Σ (targetₖ − outₖ)²' },
  { label:'δ 역전파',        expr:'δ₃ₖ = (tₖ−outₖ)·outₖ·(1−outₖ)\nδ₂ᵢ = (Σδ₃ₖ·wₖᵢ)·a₂ᵢ·(1−a₂ᵢ)' },
  { label:'Weight 업데이트', expr:'w_new = w_old + η · δ · a_prev\nb_new = b_old + η · δ' },
]

// ─── Math ─────────────────────────────────────────────────────────────────────
const sig  = x => 1 / (1 + Math.exp(-x))
const r4   = v => Math.round(v * 10000) / 10000
const r3   = v => Math.round(v * 1000)  / 1000

function mkWeights() {
  const r = () => r4((Math.random() - .5) * 2)
  return {
    w1: Array.from({length:3}, () => Array.from({length:12}, r)),
    b1: Array.from({length:3}, r),
    w2: Array.from({length:2}, () => Array.from({length:3}, r)),
    b2: Array.from({length:2}, r),
  }
}

function fwdPass(x, {w1,b1,w2,b2}) {
  const z2 = b1.map((b,i) => r4(b + w1[i].reduce((s,w,j) => s + w*x[j], 0)))
  const a2 = z2.map(z => r4(sig(z)))
  const z3 = b2.map((b,i) => r4(b + w2[i].reduce((s,w,j) => s + w*a2[j], 0)))
  const a3 = z3.map(z => r4(sig(z)))
  return {z2,a2,z3,a3}
}

function gradsOf(x, a2, a3, t, w2) {
  const d3   = a3.map((o,k) => r4((t[k]-o)*o*(1-o)))
  const d2   = a2.map((a,i) => r4(d3.reduce((s,d,k) => s+d*w2[k][i],0)*a*(1-a)))
  const loss = r4(.5 * a3.reduce((s,o,k) => s+(t[k]-o)**2, 0))
  return {d3,d2,loss}
}

function updateW({w1,b1,w2,b2}, x, a2, {d2,d3}, lr) {
  return {
    w1: w1.map((row,i) => row.map((w,j) => r4(w + lr*d2[i]*x[j]))),
    b1: b1.map((b,i)   => r4(b + lr*d2[i])),
    w2: w2.map((row,k) => row.map((w,i) => r4(w + lr*d3[k]*a2[i]))),
    b2: b2.map((b,k)   => r4(b + lr*d3[k])),
  }
}

// ─── Primitive UI ─────────────────────────────────────────────────────────────
const Card = ({children, style={}}) => (
  <div style={{background:C.bg2, borderRadius:16, border:`1px solid ${C.border}`, ...style}}>
    {children}
  </div>
)

const Lbl = ({children, style={}}) => (
  <div style={{fontSize:11, fontWeight:600, color:C.text3, letterSpacing:'1.5px', marginBottom:10, ...style}}>
    {children}
  </div>
)

const Tag = ({children, color=C.accent, bg=C.tagBg}) => (
  <span style={{background:bg, color, fontSize:10, fontWeight:600, padding:'2px 10px', borderRadius:99, letterSpacing:.5}}>
    {children}
  </span>
)

function Btn({children, onClick, disabled, variant='primary', style={}}) {
  const V = {
    primary: {bg:C.accent,   fg:'#fff',   bd:C.accent,   sh:'0 2px 10px rgba(124,111,247,.28)'},
    outline: {bg:C.bg2,      fg:C.accent,  bd:C.accent,   sh:'none'},
    ghost:   {bg:C.bg,       fg:C.text2,   bd:C.border2,  sh:'none'},
    green:   {bg:'#ecfdf5',  fg:C.green,   bd:'#86efac',  sh:'none'},
    orange:  {bg:'#fffbeb',  fg:C.orange,  bd:'#fcd34d',  sh:'none'},
    danger:  {bg:'#fff1f2',  fg:C.danger,  bd:'#fca5a5',  sh:'none'},
  }[variant]
  return (
    <button onClick={onClick} disabled={disabled} style={{
      background:V.bg, color:V.fg, border:`1.5px solid ${V.bd}`,
      borderRadius:10, padding:'9px 14px', fontSize:12, fontWeight:600,
      cursor:disabled?'not-allowed':'pointer', opacity:disabled?.4:1,
      boxShadow:V.sh, transition:'all .15s', fontFamily:'inherit',
      whiteSpace:'nowrap', width:'100%', ...style,
    }}>{children}</button>
  )
}

function PixelGrid({pixels, px=18, lit=false}) {
  return (
    <div style={{display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:2}}>
      {pixels.map((v,i) => (
        <div key={i} style={{
          width:px, height:px, borderRadius:3, flexShrink:0,
          background: v ? (lit ? C.accent : C.text) : C.bg3,
          border:`1px solid ${v ? (lit ? C.accent2 : C.border2) : C.border}`,
          transition:'background .3s',
        }}/>
      ))}
    </div>
  )
}

function Neuron({value, label, size=46, glow=false, sub}) {
  const alpha = Math.min(.12 + Math.abs(value??0)*.6, .88)
  return (
    <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:3}}>
      <div style={{
        width:size, height:size, borderRadius:'50%',
        background:`rgba(124,111,247,${alpha})`,
        border:`2px solid ${glow ? C.orange : C.accent}`,
        boxShadow: glow ? `0 0 14px ${C.orange}55` : '0 1px 4px rgba(124,111,247,.15)',
        display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
        transition:'all .4s',
      }}>
        {value !== undefined && (
          <span style={{fontSize:9, fontFamily:"'DM Mono',monospace", color:C.text, fontWeight:700}}>
            {r3(value)}
          </span>
        )}
      </div>
      {label && <span style={{fontSize:9, color:C.text3, fontFamily:"'DM Mono',monospace"}}>{label}</span>}
      {sub   && <span style={{fontSize:8, color:C.orange, fontFamily:"'DM Mono',monospace"}}>{sub}</span>}
    </div>
  )
}

function StepBar({step}) {
  return (
    <div style={{display:'flex', alignItems:'center', overflowX:'auto', gap:0}}>
      {STEP_META.map((s,i) => (
        <React.Fragment key={i}>
          <div style={{
            display:'flex', alignItems:'center', gap:5, flexShrink:0,
            padding:'5px 10px', borderRadius:99,
            background: step===i ? C.tagBg : 'transparent',
            border:`1.5px solid ${step===i ? C.accent : 'transparent'}`,
            transition:'all .25s',
          }}>
            <span style={{fontSize:11}}>{s.icon}</span>
            <span style={{
              fontSize:10, fontWeight:step===i?700:400, letterSpacing:.5,
              color: step===i ? C.accent : step>i ? '#34d399' : C.text3,
            }}>
              {step>i?'✓':`${i+1}.`} {s.label}
            </span>
          </div>
          {i < STEP_META.length-1 && (
            <div style={{flexShrink:0, width:12, height:1.5, background:step>i?C.accent:C.border, transition:'background .3s'}}/>
          )}
        </React.Fragment>
      ))}
    </div>
  )
}

const wBg = v => v>=0
  ? `rgba(124,111,247,${Math.min(.07+v*.17,.42)})`
  : `rgba(244,63,94,${Math.min(.07+Math.abs(v)*.17,.42)})`
const wFg = v => v>=0 ? '#4c3fc9' : '#be123c'

function LossCurve({history, epoch}) {
  const ref = useRef()
  useEffect(() => {
    const cv = ref.current; if(!cv) return
    const ctx = cv.getContext('2d')
    const W=cv.width, H=cv.height
    ctx.clearRect(0,0,W,H)
    ctx.fillStyle=C.bg; ctx.fillRect(0,0,W,H)
    if(history.length < 2) return
    const mx = Math.max(...history,.01)
    const pts = history.slice(-200).map((v,i,a)=>({
      x: 8+(i/(a.length-1))*(W-16),
      y: H-6-(v/mx)*(H-14),
    }))
    ctx.beginPath()
    pts.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y))
    ctx.lineTo(pts[pts.length-1].x,H-6); ctx.lineTo(pts[0].x,H-6)
    ctx.fillStyle='rgba(124,111,247,.08)'; ctx.fill()
    ctx.beginPath()
    pts.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y))
    ctx.strokeStyle=C.accent; ctx.lineWidth=1.5; ctx.stroke()
    const last=history[history.length-1]
    ctx.fillStyle=C.text2; ctx.font='9px DM Mono,Courier New'
    ctx.fillText(`loss: ${last.toFixed(5)}`,8,12)
    ctx.fillStyle=C.text3; ctx.fillText(`epoch ${epoch}`,W-68,12)
  },[history,epoch])
  return <canvas ref={ref} width={500} height={60} style={{width:'100%',borderRadius:8,display:'block'}}/>
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function BackpropVisualizerPage() {
  const [W,    setW]    = useState(mkWeights)
  const [si,   setSi]   = useState(0)
  const [lr,   setLr]   = useState(.2)
  const [step, setStep] = useState(0)
  const [fwd,  setFwd]  = useState(null)
  const [gr,   setGr]   = useState(null)
  const [hist, setHist] = useState([])
  const [ep,   setEp]   = useState(0)
  const [auto, setAuto] = useState(false)
  const [hov,  setHov]  = useState(null)
  const autoRef = useRef(false)

  const smp = SAMPLES[si]
  const inp = smp.pixels
  const tgt = smp.target

  useEffect(()=>{ setFwd(fwdPass(inp,W)); setGr(null); setStep(0) }, [si,W])

  function next() {
    if(step < 3){ setStep(s=>s+1); return }
    if(step===3){
      const g = gradsOf(inp,fwd.a2,fwd.a3,tgt,W.w2)
      setGr(g); setHist(h=>[...h.slice(-299),g.loss]); setStep(4); return
    }
    if(step===4){ setStep(5); return }
    const g = gr ?? gradsOf(inp,fwd.a2,fwd.a3,tgt,W.w2)
    setW(updateW(W,inp,fwd.a2,g,lr)); setEp(e=>e+1); setStep(0)
  }

  function train1() {
    let w=W, loss=0
    for(const s of SAMPLES){ const f=fwdPass(s.pixels,w); const g=gradsOf(s.pixels,f.a2,f.a3,s.target,w.w2); w=updateW(w,s.pixels,f.a2,g,lr); loss+=g.loss }
    setW(w); setEp(e=>e+1); setHist(h=>[...h.slice(-299),r4(loss/SAMPLES.length)]); setStep(0)
  }

  async function trainN(n) {
    autoRef.current=true; setAuto(true)
    let w=W, count=0
    while(autoRef.current && count<n){
      let loss=0
      for(const s of SAMPLES){ const f=fwdPass(s.pixels,w); const g=gradsOf(s.pixels,f.a2,f.a3,s.target,w.w2); w=updateW(w,s.pixels,f.a2,g,lr); loss+=g.loss }
      count++
      if(count%20===0){ setW({...w}); setEp(e=>e+20); setHist(h=>[...h.slice(-299),r4(loss/SAMPLES.length)]); await new Promise(r=>setTimeout(r,1)) }
    }
    setW({...w}); setEp(e=>e+count%20); autoRef.current=false; setAuto(false)
  }

  function reset() { autoRef.current=false; setAuto(false); setW(mkWeights()); setFwd(null); setGr(null); setStep(0); setHist([]); setEp(0) }

  const out      = fwd?.a3 ?? [.5,.5]
  const pred     = out[0]>out[1] ? 0 : 1
  const correct  = pred === parseInt(smp.label)
  const dispLoss = gr?.loss ?? (fwd ? r4(.5*fwd.a3.reduce((s,o,k)=>s+(tgt[k]-o)**2,0)) : null)
  const formula  = FORMULAS[Math.min(step,FORMULAS.length-1)]
  const acc      = ep===0 ? null : (() => {
    const n = SAMPLES.filter(s=>{ const f=fwdPass(s.pixels,W); return (f.a3[0]>f.a3[1]?0:1)===parseInt(s.label) }).length
    return `${n}/${SAMPLES.length}`
  })()

  // ── Layout: full-width left column + fixed right sidebar ────────────────────
  // Use a wrapper div with padding-right to carve out space for the sidebar.
  // Sidebar is position:absolute on the right so it never pushes content.
  const SIDE = 268  // sidebar width px
  const GAP  = 14

  return (
    <div style={{position:'relative'}}>

      {/* ── Sidebar (right, sticky) ── */}
      <div style={{
        position:'absolute', top:0, right:0,
        width:SIDE,
        display:'flex', flexDirection:'column', gap:14,
      }}>
        {/* Sample selector */}
        <Card style={{padding:16}}>
          <Lbl>TRAINING SAMPLE</Lbl>
          <div style={{display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:5, marginBottom:12}}>
            {SAMPLES.map((s,i)=>(
              <button key={i} onClick={()=>{setSi(i);setStep(0)}} style={{
                border:`2px solid ${si===i?C.accent:C.border}`,
                background:si===i?C.tagBg:C.bg, borderRadius:10,
                padding:'5px 3px', cursor:'pointer',
                display:'flex', flexDirection:'column', alignItems:'center', gap:3,
                transition:'all .2s',
              }}>
                <PixelGrid pixels={s.pixels} px={13} lit={si===i}/>
                <span style={{fontSize:9, color:si===i?C.accent:C.text3, fontWeight:700}}>"{s.label}"</span>
              </button>
            ))}
          </div>
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:8}}>
            <div style={{background:C.bg3, borderRadius:10, padding:'10px', textAlign:'center'}}>
              <div style={{fontSize:9, color:C.text3, marginBottom:3, letterSpacing:1}}>TARGET</div>
              <div style={{fontSize:13, fontWeight:700, color:C.accent, fontFamily:"'DM Mono',monospace"}}>[{tgt.join(', ')}]</div>
              <div style={{fontSize:10, color:C.text3, marginTop:2}}>숫자 "{smp.label}"</div>
            </div>
            <div style={{
              background:step>=2?(correct?'#ecfdf5':'#fff1f2'):C.bg3, borderRadius:10, padding:'10px', textAlign:'center',
              border:`1.5px solid ${step>=2?(correct?'#86efac':'#fca5a5'):C.border}`, transition:'all .3s',
            }}>
              <div style={{fontSize:9, color:C.text3, marginBottom:3, letterSpacing:1}}>PREDICTION</div>
              <div style={{fontSize:13, fontWeight:700, fontFamily:"'DM Mono',monospace", color:step>=2?(correct?C.green:C.danger):C.text3}}>
                {step>=2?`[${out.map(v=>r3(v)).join(', ')}]`:'[—, —]'}
              </div>
              <div style={{fontSize:10, fontWeight:600, marginTop:2, color:step>=2?(correct?C.green:C.danger):C.text3}}>
                {step>=2?(correct?'✓ Correct':'✗ Wrong'):''}
              </div>
            </div>
          </div>
        </Card>

        {/* Step control */}
        <Card style={{padding:16}}>
          <Lbl>STEP-BY-STEP</Lbl>
          <Btn onClick={next} disabled={auto} variant='primary' style={{marginBottom:8,fontSize:13}}>
            {step===5?'↺ Apply & Next Sample':`▶ Next → ${STEP_META[Math.min(step+1,5)].label}`}
          </Btn>
          <div style={{fontSize:10, color:C.text3, textAlign:'center', padding:'4px 8px', background:C.bg3, borderRadius:6}}>
            현재 <span style={{color:C.accent,fontWeight:600}}>{STEP_META[step].label}</span>
            {step<5&&<> → 다음 <span style={{color:C.text,fontWeight:500}}>{STEP_META[step+1].label}</span></>}
          </div>
        </Card>

        {/* Auto train */}
        <Card style={{padding:16}}>
          <Lbl>AUTO TRAIN</Lbl>
          <div style={{display:'flex', flexDirection:'column', gap:6}}>
            <Btn onClick={train1} disabled={auto} variant='green'>⚡ Train 1 Epoch</Btn>
            <Btn onClick={()=>trainN(100)}  disabled={auto} variant='outline'>🚀 Train 100 Epochs</Btn>
            <Btn onClick={()=>trainN(1000)} disabled={auto} variant='orange'>⚡ Train 1000 Epochs</Btn>
            {auto
              ? <Btn onClick={()=>{autoRef.current=false;setAuto(false)}} variant='danger'>⬛ Stop</Btn>
              : <Btn onClick={reset} variant='ghost'>↺ Reset Weights</Btn>
            }
          </div>
          {auto && <div style={{marginTop:8,fontSize:11,color:C.orange,textAlign:'center'}}>학습 중… epoch {ep}</div>}
        </Card>

        {/* Learning rate */}
        <Card style={{padding:16}}>
          <Lbl>LEARNING RATE η</Lbl>
          <div style={{fontSize:26,fontWeight:800,color:C.orange,marginBottom:6,fontFamily:"'DM Mono',monospace"}}>{lr.toFixed(2)}</div>
          <input type='range' min='.01' max='1' step='.01' value={lr} onChange={e=>setLr(+e.target.value)} style={{width:'100%',marginBottom:6}}/>
          <div style={{display:'flex',justifyContent:'space-between',fontSize:9,color:C.text3,marginBottom:8}}>
            <span>0.01</span><span style={{color:C.accent}}>0.2 (Excel)</span><span>1.0</span>
          </div>
          <div style={{
            fontSize:10,padding:'6px 10px',borderRadius:8,
            background:lr>.5?'#fff1f2':lr<.05?'#ecfdf5':C.tagBg,
            color:lr>.5?C.danger:lr<.05?C.green:C.accent,
            borderLeft:`2px solid ${lr>.5?C.danger:lr<.05?C.green:C.accent}`,
          }}>
            {lr>.5?'⚠ 크면 발산할 수 있습니다':lr<.05?'매우 느리게 수렴합니다':'Excel 예제 권장값 (η = 0.2)'}
          </div>
        </Card>

        {/* Formula */}
        <Card style={{padding:16}}>
          <Lbl>FORMULA · {formula.label}</Lbl>
          <div style={{background:C.bg3,borderRadius:10,padding:'12px 14px'}}>
            <pre style={{fontSize:11,color:C.accent,fontFamily:"'DM Mono',monospace",margin:0,whiteSpace:'pre-wrap',lineHeight:1.9,fontWeight:600}}>
              {formula.expr}
            </pre>
          </div>
        </Card>

        {/* Stats */}
        <Card style={{padding:16}}>
          <Lbl>STATISTICS</Lbl>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8,marginBottom:8}}>
            <div style={{background:C.bg3,borderRadius:8,padding:'8px',textAlign:'center'}}>
              <div style={{fontSize:9,color:C.text3,marginBottom:2}}>EPOCH</div>
              <div style={{fontSize:20,fontWeight:800,color:C.accent,fontFamily:"'DM Mono',monospace"}}>{ep}</div>
            </div>
            <div style={{background:C.bg3,borderRadius:8,padding:'8px',textAlign:'center'}}>
              <div style={{fontSize:9,color:C.text3,marginBottom:2}}>LOSS</div>
              <div style={{fontSize:16,fontWeight:800,fontFamily:"'DM Mono',monospace",
                color:hist.length>0&&hist[hist.length-1]<.05?C.green:C.danger}}>
                {hist.length>0?hist[hist.length-1].toFixed(4):'—'}
              </div>
            </div>
          </div>
          {acc&&(
            <div style={{
              padding:'8px 10px',borderRadius:8,textAlign:'center',fontSize:11,fontWeight:600,
              background:acc==='10/10'?'#ecfdf5':C.bg3,
              color:acc==='10/10'?C.green:C.text2,
              border:`1.5px solid ${acc==='10/10'?'#86efac':C.border}`,
            }}>
              정확도: {acc} 샘플 정답 {acc==='10/10'&&'🎉'}
            </div>
          )}
        </Card>
      </div>

      {/* ── Main content (padded right to avoid sidebar) ── */}
      <div style={{paddingRight: SIDE + GAP}}>

        {/* Header */}
        <div style={{marginBottom:20}}>
          <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:6,flexWrap:'wrap'}}>
            <h2 style={{fontSize:22,fontWeight:700,color:C.text,margin:0,letterSpacing:'-.3px'}}>
              🧠 Backprop Visualizer
            </h2>
            <Tag>Excel Lecture 07</Tag>
            <Tag bg='#ecfdf5' color={C.green}>12 → 3 → 2</Tag>
          </div>
          <p style={{fontSize:13,color:C.text2,margin:0,lineHeight:1.6}}>
            Excel 예제의 역전파 계산 과정을 웹에서 단계별로 재현합니다.
            4×3 픽셀 입력 → 은닉층 3개 → 출력층 2개 → 숫자 0 또는 1 예측.
          </p>
        </div>

        {/* Step progress */}
        <Card style={{padding:'10px 16px',marginBottom:14}}>
          <StepBar step={step}/>
        </Card>

        {/* Step description */}
        <div style={{
          background:C.tagBg,borderRadius:12,padding:'10px 16px',marginBottom:18,
          borderLeft:`3px solid ${C.accent}`,display:'flex',gap:10,alignItems:'flex-start',
        }}>
          <span style={{fontSize:17,lineHeight:1.4,flexShrink:0}}>{STEP_META[step].icon}</span>
          <div>
            <div style={{fontSize:10,fontWeight:700,color:C.accent,letterSpacing:1,marginBottom:3}}>
              STEP {step+1} · {STEP_META[step].label}
            </div>
            <p style={{fontSize:12,color:C.text,margin:0,lineHeight:1.65}}>{STEP_META[step].desc}</p>
          </div>
        </div>

        {/* Network diagram */}
        <Card style={{padding:20,marginBottom:14}}>
          <Lbl>NEURAL NETWORK  12 → 3 → 2</Lbl>
          {/* inner scroll for very narrow screens */}
          <div style={{overflowX:'auto'}}>
            <div style={{display:'flex',alignItems:'center',gap:12,minWidth:480}}>

              {/* INPUT */}
              <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:8,flexShrink:0}}>
                <div style={{fontSize:10,fontWeight:600,color:C.text3,letterSpacing:1}}>INPUT</div>
                <PixelGrid pixels={inp} px={22} lit={step>=1}/>
                <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:2,marginTop:2}}>
                  {inp.map((v,i)=>(
                    <div key={i} style={{
                      width:24,height:16,borderRadius:3,
                      background:v?(step>=1?C.tagBg:C.bg3):C.bg,
                      border:`1px solid ${v?(step>=1?C.accent:C.border2):C.border}`,
                      display:'flex',alignItems:'center',justifyContent:'center',
                      fontSize:8,color:v?C.accent:C.text3,fontWeight:700,
                      fontFamily:"'DM Mono',monospace",transition:'all .3s',
                    }}>{v}</div>
                  ))}
                </div>
                <div style={{fontSize:9,color:C.text3}}>12 pixels</div>
              </div>

              {/* → */}
              <div style={{display:'flex',flexDirection:'column',alignItems:'center',flexShrink:0}}>
                <span style={{fontSize:20,color:step>=1?C.accent:C.border,transition:'color .3s'}}>→</span>
                {step>=1&&<span style={{fontSize:8,color:C.text3,fontFamily:"'DM Mono',monospace"}}>w₁·x</span>}
              </div>

              {/* HIDDEN */}
              <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:10,flexShrink:0}}>
                <div style={{fontSize:10,fontWeight:600,color:C.text3,letterSpacing:1}}>HIDDEN</div>
                {[0,1,2].map(i=>(
                  <Neuron key={i}
                    value={step>=1&&fwd?fwd.a2[i]:undefined}
                    label={`h${i+1}`}
                    glow={step===1||step===4}
                    sub={step===4&&gr?`δ=${r3(gr.d2[i])}`:step>=1&&fwd?`z=${r3(fwd.z2[i])}`:undefined}
                  />
                ))}
                <div style={{fontSize:9,color:C.text3}}>3 nodes · sigmoid</div>
              </div>

              {/* → */}
              <div style={{display:'flex',flexDirection:'column',alignItems:'center',flexShrink:0}}>
                <span style={{fontSize:20,color:step>=2?C.accent:C.border,transition:'color .3s'}}>→</span>
                {step>=2&&<span style={{fontSize:8,color:C.text3,fontFamily:"'DM Mono',monospace"}}>w₂·a₂</span>}
              </div>

              {/* OUTPUT */}
              <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:12,flexShrink:0}}>
                <div style={{fontSize:10,fontWeight:600,color:C.text3,letterSpacing:1}}>OUTPUT</div>
                {[0,1].map(k=>{
                  const isTgt = tgt[k]===1
                  const col   = step>=2?(isTgt?C.green:C.danger):C.accent
                  return (
                    <div key={k} style={{display:'flex',flexDirection:'column',alignItems:'center',gap:3}}>
                      <div style={{
                        width:52,height:52,borderRadius:'50%',
                        background:step>=2&&fwd?`rgba(${isTgt?'5,150,105':'220,38,38'},${.1+(fwd.a3[k]??0)*.5})`:C.bg3,
                        border:`2px solid ${step>=2?col:C.border2}`,
                        boxShadow:step>=2?`0 0 14px ${col}33`:'none',
                        display:'flex',alignItems:'center',justifyContent:'center',
                        transition:'all .4s',
                      }}>
                        {step>=2&&fwd&&(
                          <span style={{fontSize:10,fontFamily:"'DM Mono',monospace",color:C.text,fontWeight:700}}>
                            {r3(fwd.a3[k])}
                          </span>
                        )}
                      </div>
                      <div style={{textAlign:'center',lineHeight:1.5}}>
                        <div style={{fontSize:9,color:C.text3,fontFamily:"'DM Mono',monospace"}}>out{k}</div>
                        <div style={{fontSize:9,fontWeight:700,color:isTgt?C.green:C.text3}}>t={tgt[k]}</div>
                        {step>=4&&gr&&<div style={{fontSize:8,color:C.orange,fontFamily:"'DM Mono',monospace"}}>δ={r3(gr.d3[k])}</div>}
                      </div>
                    </div>
                  )
                })}
                <div style={{fontSize:9,color:C.text3}}>2 nodes · sigmoid</div>
              </div>

              {/* RESULT */}
              <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:8,marginLeft:8,flexShrink:0}}>
                <div style={{fontSize:10,fontWeight:600,color:C.text3,letterSpacing:1}}>RESULT</div>
                <div style={{
                  width:64,height:64,borderRadius:'50%',
                  background:step>=2?(correct?'#ecfdf5':'#fff1f2'):C.bg3,
                  border:`3px solid ${step>=2?(correct?C.green:C.danger):C.border2}`,
                  boxShadow:step>=2?`0 0 18px ${correct?C.green:C.danger}33`:'none',
                  display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',
                  transition:'all .4s',
                }}>
                  <span style={{fontSize:22,fontWeight:800,color:step>=2?(correct?C.green:C.danger):C.text3}}>
                    {step>=2?pred:'?'}
                  </span>
                  <span style={{fontSize:9,fontWeight:600,color:step>=2?(correct?C.green:C.danger):C.text3}}>
                    {step>=2?(correct?'✓ OK':'✗ ERR'):''}
                  </span>
                </div>
                {step>=3&&dispLoss!==null&&(
                  <div style={{textAlign:'center'}}>
                    <div style={{fontSize:9,color:C.text3}}>LOSS</div>
                    <div style={{fontSize:15,fontWeight:800,fontFamily:"'DM Mono',monospace",color:dispLoss<.05?C.green:C.danger}}>
                      {dispLoss.toFixed(5)}
                    </div>
                  </div>
                )}
              </div>

            </div>
          </div>

          {/* Legend */}
          <div style={{display:'flex',gap:16,marginTop:14,flexWrap:'wrap'}}>
            {[
              {color:'rgba(124,111,247,.45)',label:'positive weight'},
              {color:'rgba(244,63,94,.45)',  label:'negative weight'},
              {color:C.orange,               label:'gradient / δ'},
              {color:C.green,                label:'target node'},
            ].map(({color,label})=>(
              <div key={label} style={{display:'flex',alignItems:'center',gap:5}}>
                <div style={{width:20,height:4,background:color,borderRadius:2}}/>
                <span style={{fontSize:10,color:C.text3}}>{label}</span>
              </div>
            ))}
          </div>
        </Card>

        {/* Weight tables */}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14,marginBottom:14}}>

          {/* W1 */}
          <Card style={{padding:16}}>
            <Lbl>
              W₁ · HIDDEN WEIGHTS <span style={{fontWeight:400,color:C.text3}}>(3×12)</span>
              {step===5&&<span style={{color:C.orange,marginLeft:6}}>← updating</span>}
            </Lbl>
            <div style={{overflowX:'auto'}}>
              <table style={{borderCollapse:'collapse',fontSize:8.5,fontFamily:"'DM Mono',monospace"}}>
                <thead>
                  <tr>
                    <th style={{padding:'2px 4px',color:C.text3,textAlign:'left',fontWeight:500}}></th>
                    {Array.from({length:12},(_,j)=>(
                      <th key={j} style={{padding:'2px 2px',color:C.text3,fontWeight:400}}>x{j+1}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {W.w1.map((row,i)=>(
                    <tr key={i}>
                      <td style={{padding:'2px 4px',color:C.accent,fontWeight:700}}>h{i+1}</td>
                      {row.map((w,j)=>{
                        const delta = gr&&step===5 ? r4(lr*gr.d2[i]*inp[j]) : null
                        const isH   = hov?.l==='w1'&&hov.i===i&&hov.j===j
                        return (
                          <td key={j}
                            onMouseEnter={()=>setHov({l:'w1',i,j,v:w,delta})}
                            onMouseLeave={()=>setHov(null)}
                            style={{
                              padding:'2px 2px',textAlign:'center',borderRadius:2,
                              background:isH?C.tagBg:wBg(w),color:wFg(w),fontWeight:600,
                              outline:step===5&&delta?`1px solid ${C.orange}`:'none',
                              transition:'background .15s',cursor:'default',minWidth:28,
                            }}>
                            {w.toFixed(2)}
                            {step===5&&delta!==null&&<div style={{fontSize:7,color:C.orange}}>{delta>0?'+':''}{delta.toFixed(3)}</div>}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{marginTop:8,display:'flex',gap:8,flexWrap:'wrap'}}>
              <span style={{fontSize:9,color:C.text3}}>bias:</span>
              {W.b1.map((b,i)=>(
                <span key={i} style={{fontSize:9,color:C.text2,fontFamily:"'DM Mono',monospace"}}>h{i+1}={b.toFixed(3)}</span>
              ))}
            </div>
          </Card>

          {/* W2 */}
          <Card style={{padding:16}}>
            <Lbl>
              W₂ · OUTPUT WEIGHTS <span style={{fontWeight:400,color:C.text3}}>(2×3)</span>
              {step===5&&<span style={{color:C.orange,marginLeft:6}}>← updating</span>}
            </Lbl>
            <table style={{borderCollapse:'collapse',fontSize:11,fontFamily:"'DM Mono',monospace",width:'100%'}}>
              <thead>
                <tr>
                  <th style={{padding:'3px 6px',color:C.text3,fontWeight:500,textAlign:'left'}}></th>
                  {['h1','h2','h3'].map(h=><th key={h} style={{padding:'3px 10px',color:C.text3,fontWeight:400}}>{h}</th>)}
                  <th style={{padding:'3px 8px',color:C.text3,fontWeight:400}}>bias</th>
                </tr>
              </thead>
              <tbody>
                {W.w2.map((row,k)=>(
                  <tr key={k}>
                    <td style={{padding:'4px 6px',color:C.accent,fontWeight:700}}>out{k}</td>
                    {row.map((w,i)=>{
                      const delta = gr&&step===5&&fwd ? r4(lr*gr.d3[k]*fwd.a2[i]) : null
                      const isH   = hov?.l==='w2'&&hov.k===k&&hov.i===i
                      return (
                        <td key={i}
                          onMouseEnter={()=>setHov({l:'w2',k,i,v:w,delta})}
                          onMouseLeave={()=>setHov(null)}
                          style={{
                            padding:'4px 10px',textAlign:'center',borderRadius:4,
                            background:isH?C.tagBg:wBg(w),color:wFg(w),fontWeight:600,
                            outline:step===5&&delta?`1px solid ${C.orange}`:'none',
                            transition:'background .15s',cursor:'default',
                          }}>
                          {w.toFixed(3)}
                          {step===5&&delta!==null&&<div style={{fontSize:9,color:C.orange}}>{delta>0?'+':''}{delta.toFixed(4)}</div>}
                        </td>
                      )
                    })}
                    <td style={{padding:'4px 8px',color:C.text2}}>
                      {W.b2[k].toFixed(3)}
                      {step===5&&gr&&<div style={{fontSize:9,color:C.orange}}>{r4(lr*gr.d3[k])>0?'+':''}{r4(lr*gr.d3[k]).toFixed(4)}</div>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {step>=4&&gr&&(
              <div style={{marginTop:10,padding:'8px 10px',background:C.bg3,borderRadius:8}}>
                <div style={{fontSize:9,color:C.orange,fontWeight:700,letterSpacing:1,marginBottom:5}}>GRADIENTS</div>
                <div style={{display:'flex',gap:10,flexWrap:'wrap'}}>
                  {gr.d3.map((d,k)=>(
                    <span key={k} style={{fontSize:10,color:C.text2,fontFamily:"'DM Mono',monospace"}}>
                      δ₃[{k}]=<strong style={{color:C.orange}}>{d.toFixed(4)}</strong>
                    </span>
                  ))}
                  {gr.d2.map((d,i)=>(
                    <span key={i} style={{fontSize:10,color:C.text2,fontFamily:"'DM Mono',monospace"}}>
                      δ₂[{i}]=<strong style={{color:C.accent}}>{d.toFixed(4)}</strong>
                    </span>
                  ))}
                </div>
              </div>
            )}
            {hov&&(
              <div style={{marginTop:8,padding:'5px 10px',background:C.tagBg,borderRadius:6,fontSize:10,color:C.text}}>
                w = <strong>{hov.v.toFixed(4)}</strong>
                {hov.delta!==null&&<span style={{color:C.orange}}> → Δ = {hov.delta.toFixed(4)}</span>}
              </div>
            )}
          </Card>
        </div>

        {/* Loss curve */}
        <Card style={{padding:16,marginBottom:14}}>
          <Lbl>LOSS CURVE</Lbl>
          <LossCurve history={hist} epoch={ep}/>
          {hist.length===0&&<div style={{fontSize:11,color:C.text3,textAlign:'center',marginTop:6}}>학습을 시작하면 Loss 그래프가 표시됩니다</div>}
        </Card>

        {/* Concept cards */}
        <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:12}}>
          {[
            {title:'왜 역전파가 필요한가?',color:C.accent,body:'Weight를 무작위로 수정하면 수백만 번의 시도가 필요합니다. 역전파는 chain rule로 각 weight가 loss에 얼마나 기여했는지를 단 한 번에 계산합니다.'},
            {title:'sigmoid의 역할',color:C.orange,body:'σ(z) = 1 / (1 + e⁻ᶻ)\n\n임의의 실수를 0~1 범위로 압축합니다. 미분이 σ(1−σ)로 간단해 역전파 계산이 효율적입니다.'},
            {title:'Weight 업데이트 원칙',color:C.green,body:'w_new = w_old + η × δ × a_prev\n\ngradient 방향으로 weight를 수정합니다. learning rate η가 수정 폭을 결정합니다.'},
          ].map(({title,color,body})=>(
            <div key={title} style={{background:C.bg2,borderRadius:12,padding:'12px 14px',border:`1px solid ${C.border}`}}>
              <div style={{fontSize:11,fontWeight:700,color,marginBottom:6}}>{title}</div>
              <pre style={{fontSize:10,color:C.text2,lineHeight:1.7,margin:0,whiteSpace:'pre-wrap',fontFamily:"'DM Mono',monospace"}}>{body}</pre>
            </div>
          ))}
        </div>

      </div>{/* end main content */}
    </div>
  )
}
