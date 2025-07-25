import { useState, useRef, useEffect } from "react";
import cx from "classnames";

import Div from "@/baseComponents/reusableComponents/Div";

import styles from "./Accordion.module.scss";

const Accordion = ({
  isActive,
  className,
  onClick,
  children,
  style = {},
  initialHeight = 0,
  animTime = 0.3,
  ...props
}) => {
  const parentRef = useRef();

  const [height, setHeight] = useState(0);
  const [updateHeight, setUpdateHeight] = useState(false);

  useEffect(() => {
    setHeight(initialHeight);
  }, [initialHeight]);

  useEffect(() => {
    if (isActive) {
      setHeight(parentRef.current.scrollHeight);
    } else {
      setHeight(initialHeight);
    }
  }, [parentRef?.current?.scrollHeight, isActive]);

  useEffect(() => {
    if (isActive) {
      setHeight(parentRef.current.scrollHeight);
    } else {
      setHeight(initialHeight);
    }
  }, [updateHeight, isActive]);

  return (
    <>
      <Div
        className={cx("of-hidden", className)}
        {...props}
        style={{ ...style, height, transition: `all ${animTime}s linear` }}
        ref={(el) => (parentRef.current = el)}
        onClick={() => {
          setUpdateHeight(true);
          if (onClick) {
            onClick();
          }
          setTimeout(() => {
            setUpdateHeight(false);
          }, animTime * 1000);
        }}
      >
        {children}
      </Div>
    </>
  );
};

export default Accordion;
