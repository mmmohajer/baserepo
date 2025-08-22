import cx from "classnames";

import Div from "@/baseComponents/reusableComponents/Div";
import Label from "@/baseComponents/formComponents/Label";

const Type1 = ({ label, val, placeHolder, isRequired = false, ...props }) => {
  return (
    <>
      <Label label={label} isRequired={isRequired} />
      <Div className="pos-rel width-per-100">
        <textarea
          {...props}
          value={val}
          placeholder={placeHolder}
          className={cx(
            "p-all-temp-3 br-rad-px-10 br-all-solid-2 m-r-temp-5 br-black width-per-100 bg-white"
          )}
          style={{ outline: "none" }}
        />
      </Div>
    </>
  );
};

export default Type1;
